# utf-8
import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # print(self.resnet)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)

        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param encoder_dim: feature size of encoded images  2048
        :param decoder_dim: size of decoder's RNN, 512
        :param attention_dim: size of the attention network, 1024
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image.(2048, 1024)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output(512,1024)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        sum(dim=1)按列求和, 列数不变,行变化
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim) (16, 196, 1024)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)(16, 1024)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, num_pixels)
        # alpha :变换到同等维度后作+后, pixel归一化值;可认为是att后所有pixels的权重值
        alpha = self.softmax(att)  # (batch_size, num_pixels)(64, 196)
        # CNN提取的image vector同lstm前一隐含层输出作att, 规则:先变换到同等维度,作+, 再同pixel乘, 求和

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


# AttentionChannel注意力模型
class AttentionChannel(nn.Module):  # 类对象
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_chan_dim, pixels_dim, dropout=0.5):  # 定义方法
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN, 512
        :param attention_chan_dim: size of the attentionchannel network, 2048
        """
        super(AttentionChannel, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_chan_dim)  # linear layer to transform encoded image.(2048, 1024)
        self.decoder_att = nn.Linear(decoder_dim, attention_chan_dim)  # linear layer to transform decoder's output.(512, 1024)
        self.out_att = nn.Linear(encoder_dim, attention_chan_dim)   # (2048, 1024)
        self.full_att = nn.Linear(pixels_dim, 1)  # linear layer to calculate values to be softmax-ed.(64, 1024, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        sum(dim=2)按行求和, 行数不变,列变化
        """
        att1 = self.encoder_att(encoder_out)  # (64, 196, 1024)
        att2 = self.decoder_att(decoder_hidden)  # (64, 1024)
        image_feature = self.out_att(encoder_out)  # (64, 196, 1024)
        att1_2 = self.dropout(self.relu(att1 + att2.unsqueeze(1)))  # (64, 196, 1024)
        att1_2 = att1_2.permute(0, 2, 1)  # (64, 1024, 196)
        att = self.full_att(att1_2).squeeze(2)  # (batch_size, pixels_dim),(64, 1024)
        alpha_chan = self.softmax(att)  # (batch_size, num_pixels),(64, 2048)(0~1)
        attention_weighted_chan_encoding = (image_feature * alpha_chan.unsqueeze(1)).sum(dim=2)  #(batch_size,num_pixels)(64,196)

        return attention_weighted_chan_encoding, alpha_chan


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, attention_chan_dim, pixels_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network, 512
        :param embed_dim: embedding size, 512
        :param decoder_dim: size of decoder's RNN, 512
        :param vocab_size: size of vocabulary, 7003
        :param encoder_dim: feature size of encoded images, 2048
        :param dropout: dropout, 0.5
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim

        self.attention_chan_dim = attention_chan_dim  # add
        self.pixels_dim = pixels_dim  # add

        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention (spatial attention) network

        # 增加 attentionchan 注意力机制   attentionchan network
        self.attentionchan = AttentionChannel(encoder_dim, decoder_dim, attention_chan_dim, pixels_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer　(7003,512)
        self.dropout = nn.Dropout(p=self.dropout)

        # decoding LSTMCell,第一层 (input_size,hidden_size)=(512+512+1024,512). Inputs: input,(h_0,c_0), Outputs: h_1,c_1
        self.decode_step = nn.LSTMCell(embed_dim + embed_dim + attention_dim, decoder_dim, bias=True)
        self.decode_step2 = nn.LSTMCell(attention_dim+decoder_dim + pixels_dim, decoder_dim, bias=True)  # decoding LSTMCell add二层

        # language lstm model
        # self.decode_step3 = nn.LSTMCell(decoder_dim + decoder_dim, decoder_dim, bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        # self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate

        self.att_div = nn.Linear(encoder_dim, attention_dim)  # linear layer to channel-dim(2048, 1024)
        self.feature_div = nn.Linear(encoder_dim, attention_dim)  # (2048, 1024)
        # self.f_beta_chan = weight_norm(nn.Linear(decoder_dim, pixels_dim))  # create a sigmoid-activated gate (512, 196) add

        # self.input = weight_norm(nn.Linear(2*decoder_dim, decoder_dim))
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)


    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size:64, decoder_dim:512)
        c = self.init_c(mean_encoder_out)  # (batch_size:64, decoder_dim:512)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        # 输入lstm中的单词是采用什么方式编码的??具体??
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size,num_pixels,encoder_dim)(64,196,2048)
        num_pixels = encoder_out.size(1)

        # local image feature
        image_global_mean = encoder_out.mean(1).to(device)  # (batch_size, encoder_dim)(160,2048)
        image_global_mean = self.feature_div(image_global_mean)  # (160,1024)
        chan_dim = image_global_mean.size(-1)  # 1024
        # print(num_pixels)
        # Sort input data by decreasing lengths; why? apparent below
        # print("caption_lengths:", caption_lengths)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)

        # print("caption_lengths:", caption_lengths)
        # print("sort_ind:", sort_ind)
        encoder_out = encoder_out[sort_ind]  # 对图像编码输出按照sort_ind顺序进行重新整合,与解码对应
        encoded_captions = encoded_captions[sort_ind]  # encoded_captions即train中的caps,按sort_ind顺序进行字幕编码
        image_global_mean = image_global_mean[sort_ind]
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim) (batch_size, 512)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas, alpha_chan
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)  # (64, 35, 7003)初始化全部为0
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)  # (64, 35, 196)
        alphas_chan = torch.zeros(batch_size, max(decode_lengths), chan_dim).to(device)  # (64, 35, 2048)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])
            # spatial attention
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            # gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            # attention_weighted_encoding = gate * attention_weighted_encoding  # (64, 2048)
            attention_weighted_encoding = self.att_div(attention_weighted_encoding)  # (64, 1024)
            # region权重方差
            region_weight_std = attention_weighted_encoding.std()
            # 增加AttentionChannel 注意力通道模型图像区域权重 (16, 196)
            attention_weighted_chan_encoding, alpha_chan = self.attentionchan(encoder_out[:batch_size_t],
                                                                              h[:batch_size_t])
            # channel权重方差
            channel_weight_std = attention_weighted_chan_encoding.std()
            # weight balancing coefficient
            region_coff = region_weight_std/(region_weight_std+channel_weight_std)
            # chan_coff = channel_weight_std / (region_weight_std + channel_weight_std)
            # gate_chan = self.sigmoid(self.f_beta_chan(h[:batch_size_t]))  # gating scalar, (batch_size_t, pixels_dim)
            # attention_weighted_chan_encoding = gate_chan * attention_weighted_chan_encoding  # (64, 196)
            # print("attention_weighted_chan_encoding.size:", attention_weighted_chan_encoding.size())  # (64, 196)
            # print(embeddings[:batch_size_t, t, :].size())
            # lstm channel输入的拼接(batch_size_t, 512+196)

            current_input1 = torch.cat([embeddings[:batch_size_t, t, :], h2[:batch_size_t],
                                        image_global_mean[:batch_size_t]], dim=1)  # LSTM的input(batch_size_t,512+512+1024)(16, 2048)
            h, c = self.decode_step(
                current_input1, (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t,decoder_dim),(batch_size_t,512)
            h = self.dropout(h)
            # weight_fusion = region_coff*attention_weighted_encoding + (1-region_coff)*attention_weighted_chan_encoding # (16, 1024)
            current_input2 = torch.cat([(1-region_coff)*attention_weighted_chan_encoding,
                                        region_coff*attention_weighted_encoding, h[:batch_size_t]], dim=1)  # (16, 1732)

            # 第一层lstmcell. (input_size,hidden_size)=(1024+512+196,512). Inputs: input,(h_0,c_0); Outputs: h_1,c_1
            h2, c2 = self.decode_step2(
                current_input2, (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t,decoder_dim),(batch_size_t,512)

            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds  # 将预测的每个单词在单词表的概率值对应存在predictions列表中,按句子长短顺序
            # print("preds:", preds.size())
            # print("batch_size_t:", batch_size_t, "--t:", t)
            # print("predictions:", predictions.size())
            # print("predictions:", predictions)
            # print("predictions[:batch_size_t, t, :]:", predictions[:batch_size_t, t, :].size(), "\n")
            alphas[:batch_size_t, t, :] = alpha  # 将alpha值保存到创建的列表中, 按句子长短顺序
            alphas_chan[:batch_size_t, t, :] = alpha_chan  # add (batch_size, max(decode_lengths), 2048)
            # print(alpha_chan.size())
        return predictions, encoded_captions, decode_lengths, alphas, alphas_chan, sort_ind
