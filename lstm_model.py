import torch
import operator
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from queue import PriorityQueue
import pdb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(204800, embed_size+128)  #resnet.fc.in_features
        self.bn = nn.BatchNorm1d(embed_size+128, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        #pdb.set_trace()    
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, agr_score_vocab_size, agr_score_embed_size, num_layers, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.agr_score_embed = nn.Embedding(agr_score_vocab_size, agr_score_embed_size)
        self.lstm = nn.LSTM(embed_size+256, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.linear2 = nn.Linear(embed_size+256, 256)  # for image feature size reduction
        self.max_seg_length = max_seq_length
        
    def forward(self, features, scanpaths, lengths, labels):
        """Decode image feature vectors and generates scanpaths."""
        #print(scanpaths)
        # pdb.set_trace()
        features1 = self.linear2(features)
        embeddings = self.embed(scanpaths)


        # pdb.set_trace()
        agr_score_embeddings = self.agr_score_embed(labels)
        agr_score_embeddings = agr_score_embeddings.permute(1,0,2)

        init_features = torch.cat((features1.unsqueeze(1), agr_score_embeddings),2)
        features1 = features1.unsqueeze(1).repeat(1,embeddings.size()[1],1)
        embeddings = torch.cat((features1, embeddings), 2)
        embeddings = torch.cat((init_features, embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 

        h0 = features.unsqueeze(0)
        c0 = features.unsqueeze(0)
        hiddens, _ = self.lstm(packed, (h0,c0))
        
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, labels, states=None):
        """Generate scanpaths for given image features using greedy search."""
        sampled_ids = []
        # print(features.size())
        features1 = self.linear2(features)
        # inputs = features.unsqueeze(1)

        # pdb.set_trace()
        agr_score_embeddings = self.agr_score_embed(labels.unsqueeze(0))
        inputs = torch.cat((features1, agr_score_embeddings),1)
        inputs = inputs.unsqueeze(1)
        h0 = features.unsqueeze(0)
        c0 = features.unsqueeze(0)
        states = (h0,c0)

        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            # pdb.set_trace()          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = torch.cat((features1, inputs), 1)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)

        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
        
    def beam_search_sample(self, features, labels, states=None):
        """Generate scanpaths for given image features using greedy search."""
        beam_width = 10
        topk = 1
        sampled_ids = []
        # print(features.size())
        features1 = self.linear2(features)
        inputs = features.unsqueeze(1)

        # pdb.set_trace()
        agr_score_embeddings = self.agr_score_embed(labels.unsqueeze(0))
        inputs = torch.cat((features1, agr_score_embeddings),1)
        inputs = inputs.unsqueeze(1)
        h0 = features.unsqueeze(0)
        c0 = features.unsqueeze(0)
        states = (h0,c0)

        decoder_hidden = states
        init_wordid = torch.LongTensor([[0]]).to(device)

        endnodes  = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, init_wordid, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(),0, node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score,idx, n = nodes.get()
            decoder_hidden = n.h

            if n.wordid.item() == 1026 and n.prevNode != None:
                endnodes.append((score,idx, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            if n.wordid == 0:
                agr_score_embeddings = self.agr_score_embed(labels.unsqueeze(0))
                inputs = torch.cat((features1, agr_score_embeddings),1)
                decoder_input = inputs.unsqueeze(1)
            else:
                inputs = self.embed(n.wordid).squeeze(1)                      # inputs: (batch_size, embed_size)
                inputs = torch.cat((features1, inputs), 1)
                decoder_input = inputs.unsqueeze(1)

            hiddens, states = self.lstm(decoder_input, decoder_hidden)
            out = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            decoder_output = nn.functional.log_softmax(out,dim=1)
            decoder_hidden = states

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score,new_k+torch.rand(1), node))

            # put them into queue
            for i in range(len(nextnodes)):
                score,idx, n1 = nextnodes[i]
                nodes.put((score,idx, n1))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        
        for score,idx, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.squeeze(0))
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.squeeze(0))

            # pdb.set_trace()
            utterance = utterance[::-1]
            utterance = torch.stack(utterance,1)
        return utterance

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
