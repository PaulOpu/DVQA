class QUES(nn.Module):

    def __init__(
            self, n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, dropout_rate=0.5, mlp_hidden_size=1024):
        super(QUES, self).__init__()
        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        # self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(lstm_hidden, mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

        # self.fine_tune()  # define which parameter sets are to be fine-tuned

    def forward(self, image, question, question_len):
        #  https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        # print(embed_pack)
        output, (h, c) = self.lstm(embed_pack)
        # h_tile = h.permute(1, 0, 2).expand(
        #     batch_size, n_pair * n_pair, self.lstm_hidden
        # )

        # _, (h, c) = self.lstm(question)

        # input(h.shape)
        # out = self.mlp(h.squeeze(0))

        # Extract the outputs for the last timestep of each example

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)
        # if output.is_cuda:
        #     idx = idx.cuda(output.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        last_output = output.gather(
            time_dimension, idx).squeeze(time_dimension)
        # input(last_output.shape)
        return self.mlp(last_output)

        # emb_len = np.array(question_len)
        # sorted_idx = np.argsort(-emb_len)
        # embed = embed[sorted_idx]
        # emb_len = emb_len[sorted_idx]
        # unsorted_idx = np.argsort(sorted_idx)
        #
        # packed_emb = torch.nn.utils.rnn.pack_padded_sequence(embed, emb_len, batch_first=True)
        # output, hn = self.rnn(packed_emb)
        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
        # unpacked = unpacked.transpose(0, 1)
        # unpacked = unpacked[torch.LongTensor(unsorted_idx)]
        # return unpacked
        #
        #
        # return out


class SANDY(nn.Module):
    '''
    Implementation of SANDY based on the SANVQA model

    '''

    def __init__(
            self,
            n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, encoded_image_size=7, conv_output_size=2048,
            mlp_hidden_size=1024, dropout_rate=0.5, glimpses=2

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(SANDY, self).__init__()
        #SANDY: add 30 for the n_class
        self.n_class = n_class + 30
        #SANDY: add 30 to the n_vocab
        self.embed = nn.Embedding(n_vocab + 30, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.enc_image_size = encoded_image_size
        # resnet = torchvision.models.resnet152(
        #     pretrained=True)
        resnet = torchvision.models.resnet101(
            pretrained=True)  # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling
        modules = list(resnet.children())[:-2]
        # modules = list(resnet.children())[:-1]  # including AvgPool2d, 051019 afternoon by Xin

        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.attention = Attention(conv_output_size, lstm_hidden, mid_features=512, glimpses=glimpses, drop=0.5)

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(conv_output_size * glimpses + lstm_hidden,
                                           mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

        self.fine_tune()  # define which parameter sets are to be fine-tuned
        self.hop = 1

    def forward(self, image, question, question_len):  # this is an image blind example (as in section 4.1)
        conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)

        # normalize by feature map, why need it and why not??
        # conv_out = conv_out / (conv_out.norm(p=2, dim=1, keepdim=True).expand_as(
        #     conv_out) + 1e-8)  # Section 3.1 of show, ask, attend, tell

        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        # conv_out = conv_out.view(conv_out.size(0), -1).contiguous()

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        lstm_output, (h, c) = self.lstm(embed_pack)

        # pad packed sequence to get last timestamp of lstm hidden
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), lstm_output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)

        lstm_final_output = lstm_output.gather(
            time_dimension, idx).squeeze(time_dimension)

        attention = self.attention(conv_out, lstm_final_output)
        weighted_conv_out = apply_attention(conv_out,
                                            attention)  # (n, glimpses * channel) ## Or should be (n, glimpses * channel, H, W)?
        # augmented_lstm_output = (weighted_conv_out + lstm_final_output)
        augmented_lstm_output = torch.cat((weighted_conv_out, lstm_final_output), 1)

        if self.hop == 2:
            raise NotImplementedError
            # attention = self.attention(conv_out, lstm_final_output)
            # weighted_conv_out = apply_attention(conv_out, attention)
            # augmented_lstm_output = (weighted_conv_out + lstm_final_output)

        return self.mlp(augmented_lstm_output)

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
        else:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

        for p in self.mlp.parameters():
            p.requires_grad = True

        for p in self.attention.parameters():
            print(p.requires_grad, "p.requires_grad")
            p.requires_grad = True


def train_YES(epoch):  # basically trim over loss calculation and back prop for YES model calculation
    raise NotImplementedError


def valid_YES(epoch, val_split="val_easy",
              load_image=True):  # basically trimmed version of train() and valid(), except without image_loading and GPU tensor operation
    raise NotImplementedError

class YES(nn.Module):
    def __init__(
            self,
            n_class, yes_class_idx,

    ):
        super(YES, self).__init__()
        self.n_class = n_class
        self.yes_class_idx = yes_class_idx
        self.f = nn.Sequential(
            nn.Linear(10, 3),
        )

    def forward(self, image, question, question_len):
        result = np.zeros([image.shape[0], self.n_class])
        result[:, self.yes_class_idx] = 1
        return torch.Tensor(result)


class IMG(nn.Module):
    def __init__(
            self,
            n_class, encoded_image_size=7, conv_output_size=2048, mlp_hidden_size=1024, dropout_rate=0.5

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(IMG, self).__init__()
        self.n_class = n_class

        self.enc_image_size = encoded_image_size
        # resnet = torchvision.models.resnet101(
        #     pretrained=True)  # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        resnet = _resnet101(pretrained=True)
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling
        modules = list(resnet.children())[:-2]
        # modules = list(resnet.children())[:-1]  # including AvgPool2d, 051019 afternoon by Xin

        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.mlp = nn.Sequential(
            self.dropout,
            nn.Linear(conv_output_size * encoded_image_size * encoded_image_size, mlp_hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(mlp_hidden_size, self.n_class))

        # self.mlp = nn.Sequential(self.dropout,
        #                          nn.Linear(conv_output_size, mlp_hidden_size),
        #                          nn.ReLU(),
        #                          self.dropout,
        #                          nn.Linear(mlp_hidden_size, self.n_class))  # including AvgPool2d, 051019 afternoon by Xin

        self.fine_tune()  # define which parameter sets are to be fine-tuned

    def forward(self, image, question, question_len):  # this is an image blind example (as in section 4.1)
        conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)
        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        final_out = self.mlp(conv_out.view(conv_out.size(0), -1).contiguous())

        return final_out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
            # for c in list(self.resnet.children())[6:]:
            #     for p in c.parameters():
            #         p.requires_grad = fine_tune
        else:
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4.
            # Before that not-fine-tuned are block 1 and preliminary blocks.
            # >>> print(len(list(resnet.children()))) >>> 10 -> we only utilize up to the 8th block.
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

        for p in self.mlp.parameters():
            p.requires_grad = True

        # # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        # for c in list(self.resnet.children())[5:]:
        #     for p in c.parameters():
        #         p.requires_grad = fine_tune


class IMGQUES(nn.Module):
    def __init__(
            self,
            n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, encoded_image_size=7, conv_output_size=2048,
            mlp_hidden_size=1024, dropout_rate=0.5

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(IMGQUES, self).__init__()
        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet101(
            pretrained=True)  # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling
        modules = list(resnet.children())[:-2]
        # modules = list(resnet.children())[:-1]  # including AvgPool2d, 051019 afternoon by Xin

        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(conv_output_size * encoded_image_size * encoded_image_size + lstm_hidden,
                                           mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

        self.fine_tune()  # define which parameter sets are to be fine-tuned

    def forward(self, image, question, question_len):  # this is an image blind example (as in section 4.1)
        conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)
        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        conv_out = conv_out.view(conv_out.size(0), -1).contiguous()

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        # print(embed_pack)
        output, (h, c) = self.lstm(embed_pack)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)

        last_output = output.gather(
            time_dimension, idx).squeeze(time_dimension)

        conv_lstm_feature = torch.cat((conv_out, last_output), 1)

        return self.mlp(conv_lstm_feature)

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
            # for c in list(self.resnet.children())[6:]:
            #     for p in c.parameters():
            #         p.requires_grad = fine_tune
        else:
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4.
            # Before that not-fine-tuned are block 1 and preliminary blocks.
            # >>> print(len(list(resnet.children()))) >>> 10 -> we only utilize up to the 8th block.
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

        for p in self.mlp.parameters():
            p.requires_grad = True
class SANVQA(nn.Module):
    '''
    We implement SANVQA based on https://github.com/Cyanogenoid/pytorch-vqa.
    A SAN implementation for show, ask, attend and tell

    Currently as one-hop
    TODO: change to two-hops
    '''

    def __init__(
            self,
            n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, encoded_image_size=7, conv_output_size=2048,
            mlp_hidden_size=1024, dropout_rate=0.5, glimpses=2

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(SANVQA, self).__init__()
        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.enc_image_size = encoded_image_size
        # resnet = torchvision.models.resnet152(
        #     pretrained=True)

        
          # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling

        
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.attention = Attention(conv_output_size, lstm_hidden, mid_features=512, glimpses=glimpses, drop=0.5)

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(conv_output_size * glimpses + lstm_hidden,
                                           mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

          # define which parameter sets are to be fine-tuned
        self.hop = 1

        act_f = nn.ReLU()

        resnet = torchvision.models.resnet101(
            pretrained=True)
            
        #modules = list(resnet.children())[:-2]
        #EarlyConcat: modules = list(resnet.children())[:5]
        concat_point = 6
        modules = list(resnet.children())[:concat_point]

        #self.resnet = nn.Sequential(*modules)
        self.resnet = nn.Sequential(*modules)

        #Chargrid: Early Concat
        untrained_resnet = torchvision.models.resnet101(
            pretrained=False)

        #EarlyConcat: modules = list(untrained_resnet.children())[:5]
        chargrid_modules = list(untrained_resnet.children())[:concat_point]
        #replace first module to fit dimensions
        chargrid_modules[0:1] = [
            nn.Conv2d(45, 10, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(10),
            act_f,
            nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3)]

        self.chargrid_net = nn.Sequential(*chargrid_modules)

        #After Concat

        #EarlyConcat: entitygrid_modules = list(untrained_resnet.children())[5:-2]
        entitygrid_modules = list(untrained_resnet.children())[concat_point:-2]
        #concatentation doubles the number of channels (512->1024)
        entitygrid_modules[0][0].conv1 = nn.Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        entitygrid_modules[0][0].downsample[0] = nn.Conv2d(
            1024, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)

        self.entitygrid_net = nn.Sequential(*entitygrid_modules)
        
        
        self.fine_tune(True)

        #Chargrid: Network before concat with image (224/112/56/28/14)
        """ self.chargrid_net = nn.Sequential(
                nn.Conv2d(45, 10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                act_f,
                nn.Conv2d(10, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                act_f,
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                act_f,
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                act_f,
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                act_f
            )

        self.entitygrid_net = nn.Sequential(
                #nn.Conv2d(3072, 2048, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(2048),
                #act_f,
                nn.Conv2d(3072, 2048, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2048),
                act_f,
                nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                act_f,
                nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(2048),
                act_f,
                nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2048),
                act_f
            ) """




    def forward(self, image, question, question_len, chargrid):  # this is an image blind example (as in section 4.1)
        #conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)
        conv_out = self.resnet(image)

        # normalize by feature map, why need it and why not??
        # conv_out = conv_out / (conv_out.norm(p=2, dim=1, keepdim=True).expand_as(
        #     conv_out) + 1e-8)  # Section 3.1 of show, ask, attend, tell

        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        # conv_out = conv_out.view(conv_out.size(0), -1).contiguous()

        #Chargrid: Enlarge Img Vector
        #conv_out = enlarge_batch_tensor(conv_out)
        chargrid = self.chargrid_net(chargrid)

        conv_out = torch.cat([conv_out,chargrid],dim=1)
        conv_out = self.entitygrid_net(conv_out)

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        lstm_output, (h, c) = self.lstm(embed_pack)

        # pad packed sequence to get last timestamp of lstm hidden
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), lstm_output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)

        lstm_final_output = lstm_output.gather(
            time_dimension, idx).squeeze(time_dimension)

        attention = self.attention(conv_out, lstm_final_output)
        weighted_conv_out = apply_attention(conv_out,
                                            attention)  # (n, glimpses * channel) ## Or should be (n, glimpses * channel, H, W)?
        # augmented_lstm_output = (weighted_conv_out + lstm_final_output)
        augmented_lstm_output = torch.cat((weighted_conv_out, lstm_final_output), 1)

        if self.hop == 2:
            raise NotImplementedError
            # attention = self.attention(conv_out, lstm_final_output)
            # weighted_conv_out = apply_attention(conv_out, attention)
            # augmented_lstm_output = (weighted_conv_out + lstm_final_output)

        return self.mlp(augmented_lstm_output)

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
        else:
            for c in list(self.resnet.children()):
            #for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
            

        for p in self.mlp.parameters():
            p.requires_grad = True

        for p in self.attention.parameters():
            print(p.requires_grad, "p.requires_grad")
            p.requires_grad = True

        #Chargrid and EntityGrid
        for c in list(self.chargrid_net.children()):
            for p in c.parameters():
                p.requires_grad = True
        for c in list(self.entitygrid_net.children()):
            for p in c.parameters():
                p.requires_grad = True

