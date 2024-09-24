import torch
import matplotlib.pyplot as plt
import numpy as np 
import math
import argparse
import pickle 
import os
from torchvision import transforms 
from testimg_loader import get_loader 
from lstm_model import EncoderCNN, DecoderRNN
from PIL import Image
import pdb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    feat_x = args.image_sz[0]/args.patch_sz[0]
    feat_y = args.image_sz[1]/args.patch_sz[1]
    total_patches = feat_x * feat_y
    BOS = int(total_patches+1)
    EOS = int(total_patches+2)
    vocab_sz = int(total_patches+3)

    # To be used in coversion of sequence of image patches to sequence of fixations
    classID_map = np.zeros([args.image_sz[0],args.image_sz[1]])
    for i in range(1,args.image_sz[0]+1):
        for j in range(1,args.image_sz[1]+1):
            classID_map[i-1,j-1] = math.ceil(i/16)+(math.ceil(j/16)-1)*feat_x

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    test_data_loader = get_loader(args.image_path, transform, args.image_sz, args.patch_sz,
                             shuffle=False)
    
    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, args.vocab_length, args.agr_score_vocab_length, args.agr_score_embed_size, args.num_layers).eval()

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    curr_epoch=299

    for epoch in range(curr_epoch, args.num_epochs):
            snapshot_name = 'epoch_%d' % (epoch)
            print(snapshot_name)

            # Load the trained model parameters
            encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder' + snapshot_name + '.ckpt')))
            decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'decoder' + snapshot_name + '.ckpt')))

            total_step = len(test_data_loader)

            # Prepare an image
            for i, (images,image_basename, img_org_sz) in enumerate(test_data_loader): 

                cuda = torch.device("cuda:0")
                image_tensor = images.to(device)

                # Generate a scanpath from the image
                feature = encoder(image_tensor)
                size_feature = feature.size()

                image_name = os.path.splitext(image_basename[0])[0]

                if args.mode == 0:

                    start_label = 3
                    last_label = 12
                    sampled_ids = []
                    # class_seqID_BS_img = np.zeros((last_label,30))
                    for labels in range(start_label,last_label+1):
                        
                        labels = torch.tensor(labels).to(device)
                        z = torch.FloatTensor(np.random.normal(0, 1, (size_feature[0], 128))).to(device)
                        features = torch.cat((feature,z),1)

                        #for max sampling ----------------------------------------------
                        sampled_ids = decoder.sample(features, labels)
                        sampled_ids = sampled_ids[0].cpu().numpy()        
                        # print(sampled_ids)
                        scanpath = classIDtotraj(classID_map, sampled_ids, vocab_sz, BOS, EOS, feat_x, feat_y, args.image_sz, img_org_sz)
                        np.save(f'./Results/mode{args.mode}/{image_name}_{labels}.npy', scanpath)
                        
                        # #-------------------------------------

                        # # for beam search sampling--------------------------------------
                        # sampled_ids = decoder.beam_search_sample(features, labels)
                        # sampled_ids = sampled_ids[0].cpu().numpy()  
                        # print(sampled_ids)
                        # for j in range(1,sampled_ids.shape[0]):
                        #     class_seqID_BS_img[labels-1,j-1] = sampled_ids[j]
                        # class_seqID_BS_img[labels-1,sampled_ids.shape[0]-1] = 1026
                        # #-------------------------------------

                if args.mode == 1:

                    label = 9
                    sampled_ids = []
                    # class_seqID_BS_img = np.zeros((last_label,30))
                    for trial in range(10):
                        
                        labels = torch.tensor(label).to(device)
                        z = torch.FloatTensor(np.random.normal(0, 1, (size_feature[0], 128))).to(device)
                        features = torch.cat((feature,z),1)

                        #for max sampling ----------------------------------------------
                        sampled_ids = decoder.sample(features, labels)
                        sampled_ids = sampled_ids[0].cpu().numpy()        
                        # print(sampled_ids)
                        scanpath = classIDtotraj(classID_map, sampled_ids, vocab_sz, BOS, EOS, feat_x, feat_y, args.image_sz, img_org_sz)
                        np.save(f'./Results/mode{args.mode}/{image_name}_{trial}.npy', scanpath)
                        
                        # #-------------------------------------

                        # # for beam search sampling--------------------------------------
                        # sampled_ids = decoder.beam_search_sample(features, labels)
                        # sampled_ids = sampled_ids[0].cpu().numpy()  
                        # print(sampled_ids)
                        # for j in range(1,sampled_ids.shape[0]):
                        #     class_seqID_BS_img[labels-1,j-1] = sampled_ids[j]
                        # class_seqID_BS_img[labels-1,sampled_ids.shape[0]-1] = 1026
                        # #-------------------------------------

def classIDtotraj(classID_map, sampled_ids, vocab_size, BOS, EOS, feat_x, feat_y, img_sz, img_org_sz):

    # seq_len = sampled_ids.shape[0]
    last_pts = np.where(sampled_ids == EOS)
    if last_pts[0].size == 0 :
        last_ptID = 20
    else:
        last_ptID = last_pts[0][0]
    traj_classID =  sampled_ids[0:last_ptID]
    FP_loc=np.where(traj_classID  == BOS)
    traj_classID = np.delete(traj_classID,FP_loc)
    FP_loc=np.where(traj_classID  == 0)
    traj_classID = np.delete(traj_classID,FP_loc)

    # print(traj_classID)  
    seq_len = traj_classID.shape[0]        
    scanpath = np.zeros((3,seq_len))
    for ind in range(len(traj_classID)):
        class_id = traj_classID[ind]
        locations = np.where(classID_map == class_id)
        x = locations[0][127]
        y = locations[1][7]
        scanpath[0,ind] = (x*int(img_org_sz[0]))/img_sz[0]
        scanpath[1,ind] = (y*int(img_org_sz[1]))/img_sz[1]

    return scanpath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--image_path', type=str,default = '../LSTM_cell_OSIE_label_noise_ROI_IOR/data/test/', help='input image for generating scanpath')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of fixation embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--vocab_length', type=int, default=1027)
    parser.add_argument('--agr_score_embed_size', type=int , default=256, help='dimension of agreement score vectors')
    parser.add_argument('--agr_score_vocab_length', type=int, default=15)
    parser.add_argument('--patch_sz', type=int, default=[16, 16])
    parser.add_argument('--image_sz', type=int, default=[512, 512])
    parser.add_argument('--mode', type=int, default=0, help='select mode 0 for generating 10 scanpaths of different varieties or 1 for 10 scanpaths of same variety') 
    args = parser.parse_args()
    main(args)
