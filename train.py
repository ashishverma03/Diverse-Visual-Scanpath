import argparse
import torch
import torch.nn as nn
import numpy as np
import math
import os
import pickle
from train_data_loader import get_loader
from testimg_loader import get_loader as val_get_loader 
from lstm_model import EncoderCNN, DecoderRNN, AverageMeter
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import pdb

# Device configuration
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu') #'cuda' if torch.cuda.is_available() else

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet      transforms.RandomCrop(args.crop_size)     transforms.RandomHorizontalFlip(),
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Build data loader
    train_data_loader = get_loader(args.train_path,
                             transform, args.image_sz, args.patch_sz, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    val_data_loader = val_get_loader(args.val_path,
                             transform, args.image_sz, args.patch_sz, shuffle=False)  

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, args.vocab_length, args.agr_score_vocab_length, args.agr_score_embed_size, args.num_layers).to(device)

    # # # Uncomment below to Load previous trained model
    # encoder.load_state_dict(torch.load(args.encoder_path))
    # decoder.load_state_dict(torch.load(args.decoder_path))

    # Continue training from epoch number
    curr_epoch = 0
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    
    for epoch in range(curr_epoch,args.num_epochs):
        train(train_data_loader, encoder, decoder, criterion, optimizer, epoch, args)
        validate(val_data_loader, encoder, decoder, optimizer, epoch, args)

def train(train_data_loader, encoder, decoder, criterion, optimizer, epoch, args):
        encoder.train()
        decoder.train()

        device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
        train_loss = AverageMeter()
        total_step = len(train_data_loader)
        Avg_train_loss = np.zeros((1,total_step))

        for i, (images, scanpaths, lengths, labels) in enumerate(train_data_loader):

            images = images.to(device)
            agr_score_label = []
            agr_score_label.append(labels)
            agr_score_label = torch.tensor(agr_score_label).to(device)
            scanpaths = scanpaths.to(device)
            targets = pack_padded_sequence(scanpaths, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            size_features = features.size()
            z = torch.FloatTensor(np.random.normal(0, 1, (size_features[0], 128))).to(device)
            features = torch.cat((features,z),1)
            outputs = decoder(features, scanpaths, lengths, agr_score_label)

            loss = criterion(outputs, targets)
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.data[0])

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, train_loss.avg, np.exp(loss.item()))) 
                
        
            Avg_train_loss[0,i] = train_loss.avg.cpu().numpy()
        
        snapshot_name = 'epoch_%d' % (epoch)
        # np.save('Avg_train_loss'+ snapshot_name, Avg_train_loss)    
            # Save the model checkpoints
            # if (i+1) % args.save_step == 0:
            #     torch.save(decoder.state_dict(), os.path.join(
            #         args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            #     torch.save(encoder.state_dict(), os.path.join(
            #         args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

def validate(val_data_loader, encoder, decoder, optimizer, epoch, args):
        encoder.eval()
        decoder.eval()

        # Save the model checkpoints
        if (epoch+1) % args.save_step == 0:

            snapshot_name = 'epoch_%d' % (epoch)
            torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder' + snapshot_name + '.ckpt'))
            torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder' + snapshot_name + '.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--encoder_path', type=str, default='/home/ashish/Documents/Codes/LSTM_cell_OSIE_label_noise_ROI_IOR_github/models/encoderepoch_299.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/home/ashish/Documents/Codes/LSTM_cell_OSIE_label_noise_ROI_IOR_github/models/decoderepoch_299.ckpt', help='path for trained decoder')

    parser.add_argument('--train_path', type=str, default='./dataset/train/', help='path for train annotation mat file')
    parser.add_argument('--val_path', type=str, default='./dataset/val/', help='path for train annotation mat file')
    parser.add_argument('--log_step', type=int , default=100, help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=20, help='number of epochs for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of location embedding vectors')
    parser.add_argument('--agr_score_embed_size', type=int , default=256, help='dimension of agreement score vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--vocab_length', type=int, default=1027)
    parser.add_argument('--agr_score_vocab_length', type=int, default=15)
    parser.add_argument('--patch_sz', type=int, default=[16, 16])
    parser.add_argument('--image_sz', type=int, default=[512, 512])
    args = parser.parse_args()
    print(args)
    main(args)
