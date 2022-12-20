import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.  # 0-1
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img).to(device)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    seqs_betas = torch.ones(k, 1, 1).to(device)  # add score
    seqs_alpha_var = torch.ones(k, 1, 1).to(device)  # add score

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    complete_seqs_betas = list()  # add score
    complete_seqs_alpha_var = list()  # alpha_var
    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        # print("sum:", torch.sum(alpha, dim=1))

        beta_t = torch.mean(awe, 1, True)  # # add visualized weight score; 1表对行求平均,0表对列求平均
        alpha_var = torch.var(alpha, 1, True).unsqueeze(1)
        print("alpha_var:", alpha_var)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe  # (s, encoder_dim), encoder_dim = 2048
        awe = decoder.con_channel(awe)  # (64, 512)

        # beta_t = awe[:, -5].unsqueeze(1)  ####### add visualized weight score
        # add visualized weight score, beta_t
        # beta_t = torch.mean(awe, 1, True)  # torch.mean(awe, 1, True) 1表对行求平均,0表对列求平均

        # 针对attentionchan(channel attention), awe_chan为attention_weighted_chan_encoding缩写
        awe_chan, alpha_chan = decoder.attentionchan(encoder_out, h)  # (s, 196), (s, 2048),decoder为实例化得到的对象
        # alpha_chan = alpha_chan.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        # gate_chan = decoder.sigmoid(decoder.f_beta_chan(h2))  # gating scalar, (s, 196)
        # awe_chan = gate_chan * awe_chan  # (s, 196)

        # beta_t = torch.mean(awe, 1, True) # add visualized weight score,

        h, c = decoder.decode_step(torch.cat([embeddings, awe, awe_chan], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(decoder.dropout(h))  # (s, vocab_size)
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

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)
        seqs_betas = torch.cat([seqs_betas[prev_word_inds], beta_t[prev_word_inds].unsqueeze(1)], dim=1)  # add score
        seqs_alpha_var = torch.cat([seqs_alpha_var[prev_word_inds], alpha_var[prev_word_inds].unsqueeze(1)], dim=1)  #add score

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

            complete_seqs_betas.extend(seqs_betas[complete_inds])  # add score
            complete_seqs_alpha_var.extend(seqs_alpha_var[complete_inds])  # add score

        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]

        seqs_betas = seqs_betas[incomplete_inds]  # add score
        seqs_alpha_var = seqs_alpha_var[incomplete_inds]  # var

        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]

        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    print("seqmax:", seq)
    alphas = complete_seqs_alpha[i]

    betas = complete_seqs_betas[i]  # add score
    vars = complete_seqs_alpha_var[i]

    print("betas", betas)
    print("vars", vars)

    return seq, alphas, betas, vars


def visualize_att(image_path, seq, alphas, betas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(80, -40, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=10)
        plt.text(80, 400, '%.2f' % (betas[t].item()), color='green', backgroundcolor='white', fontsize=8)  # add score

        plt.imshow(image)

        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.brg)  # jet: 显著色彩; # https://blog.csdn.net/weixin_40683253/article/details/87370127
        plt.axis('off')
        # plt.colorbar()      #
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    ########################################################################
    # debug时，直接赋值args = parser.parse_args( )
    args.img = './img/COCO_val2014_000000000395.jpg'
    args.model = './BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'
    args.word_map = '../chanspaiok5flickr/captiondata/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json'
    args.beam_size = 3  # 5
    print(args)
    #########################################################################

    # Load model
    checkpoint = torch.load(args.model)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas, betas, vars = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, betas, rev_word_map, args.smooth)
