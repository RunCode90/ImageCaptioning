import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval

# Parameters PATH
# data_folder = '../chanspaiok5flickr/captiondata'  # folder with data files saved by create_input_files.py
data_folder = '../../../../code/Code/chanspai-coco5/captiondata/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = './BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = '../../../../code/Code/chanspai-coco5/captiondata/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model, checkpoint模型的权重，保存当前实验状态所需的信息，以便可以从这一点恢复训练
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

nlgeval = NLGEval()  # loads the evaluator

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        infinite_pred = False  # max()出现空情况的状态标志位

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        image_global_mean = encoder_out.mean(1).to(device)  # (batch_size, encoder_dim)(160,2048)
        image_global_mean = decoder.feature_div(image_global_mean)  # (160,1024)
        chan_dim = image_global_mean.size(-1)  # 1024
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)
        h2, c2 = decoder.init_hidden_state(encoder_out)
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim), 每次输入lstmcell单词的编码
            # 针对attention(spatial attention), awe为attention_weighted_encoding缩写,第二层隐函层输出进行spatial attention.
            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels),decoder为实例化得到的对象
            # gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim) (s, 2048)
            # awe = gate * awe  # (s, 2048)
            awe = decoder.att_div(awe)  # (64, 1024)
            region_weight_std = awe.std()
            # 针对attentionchan(channel attention), awe_chan为attention_weighted_chan_encoding缩写,
            # 第一层隐函层输出进行channel attention
            awe_chan, _ = decoder.attentionchan(encoder_out, h)  # (s, 196), (s, 2048),decoder为实例化得到的对象
            # gate_chan = decoder.sigmoid(decoder.f_beta_chan(h))  # gating scalar, (s, 196)
            # awe_chan = gate_chan * awe_chan  # (s, 196)
            channel_weight_std = awe_chan.std()
            region_coff = region_weight_std / (region_weight_std + channel_weight_std)
            # decoding LSTMCell,第一层, Inputs: input,(h_0,c_0)=(512+512+196,512); Outputs: h_1,c_1
            h, c = decoder.decode_step(torch.cat([embeddings, h2, image_global_mean], dim=1), (h, c))  #(s, decoder_dim)(s, 512)
            # h1 = decoder.dropout(h)  # 第一层输出dropout
            
            # h2_0, c2_0为lstmcell第二层图像编码的初始化输入; h2, c2为lstmcell第二层的隐含层输出
            # 第二层, Inputs: input,(h2_0,c2_0)=(512+2048,512)
            current_input2 = torch.cat([(1-region_coff)*awe_chan, region_coff*awe, h], dim=1)  # (16, 1732)

            # 第一层lstmcell. (input_size,hidden_size)=(1024+512+196,512). Inputs: input,(h_0,c_0); Outputs: h_1,c_1
            h2, c2 = decoder.decode_step2(current_input2, (h2, c2))  # (batch_size_t,decoder_dim),(batch_size_t,512)

            scores = decoder.fc(decoder.dropout(h2))  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences 将当前句子和生成的下一个单词拼接在一起
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]  # 每次没有生成完整序列单词的索引号
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))  # 每次生成完整序列单词的索引号

            # Set aside complete sequences
            if len(complete_inds) > 0:  # 如有某个生成的序列完整(最后一个单词为<end>,表示生成序列完整)
                complete_seqs.extend(seqs[complete_inds].tolist())  # 将生成完整的那个序列索引号存到complete_seqs列表中
                complete_seqs_scores.extend(top_k_scores[complete_inds])  # 将生成完整的那个序列累计单词得分存com_se_ores列表中
            k -= len(complete_inds)  # reduce beam length accordingly 从总要成的句子中减去生成的那个完整句子,得到还未生成句子个数

            # Proceed with incomplete sequences
            if k == 0:  # 如果要生成的全部句子都生成结束了
                break
            seqs = seqs[incomplete_inds]  # 没有生成完整序列的句子

            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]

            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            image_global_mean = image_global_mean[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                infinite_pred = True
                break
            step += 1

        # max()出现空的情况
        if infinite_pred is not True:
            i = complete_seqs_scores.index(max(complete_seqs_scores))  # 在生成所有完整句子中,得分最大的那个索引号
            seq = complete_seqs[i]  # 得分最大的那个索引号对应的句子
        else:
            seq = seqs[0][:20]  # 取前25个word
            seq = [seq[i].item() for i in range(len(seq))]

        # i = complete_seqs_scores.index(max(complete_seqs_scores))  # 在生成所有完整句子中,得分最大的那个索引号
        # seq = complete_seqs[i]  # 得分最大的那个索引号对应的句子

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if
                           w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        img_caps = [' '.join(c) for c in img_captions]
        # print(img_caps)
        references.append(img_caps)

        # Hypotheses
        hypothesis = (
        [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypothesis = ' '.join(hypothesis)
        # print(hypothesis)
        hypotheses.append(hypothesis)
        assert len(references) == len(hypotheses)

    # Calculate scores
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    return metrics_dict


if __name__ == '__main__':
    beam_size = 5  # 采用Beam Search策略预测词汇
    metrics_dict = evaluate(beam_size)
    # print("\nscore @ beam size of %d is %.4f." % (beam_size, metrics_dict))
    print(metrics_dict)
