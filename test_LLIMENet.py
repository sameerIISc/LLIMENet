def test_LLIMENet(dataset='SONY', fraction_data='100%'):
    
    """
    This is the test code for LLIMENet. It restores low light images 
    by using pre-trained models on various datasets.   
    
    Arguments: It accepts the dataset and fraction_data as the arguments. 
    The dataset argument chooses the dataset on which the user wnats to test
    the LLIMENet on. The options are 'SONY', 'FUJI' or 'LOL'.
    The fraction_data chooses the pre-trained model that has been trained 
    on the specified fraction of the training data. The optionns are
    '10%', '10%_augmented' and '100%'. 
    
    The results will be saved in a directory (in the directory containing the codes) 
    whose name will have dataset and fraction of the data.  
    """
    
    import os
    import numpy as np
    import matplotlib as mp
    import torch
    from torch.autograd import Variable
    from BuildGPyr import BuildGPyr
    from PIL import Image
    
    cuda = torch.device('cuda:0')
    
    short_dir = os.path.join(os.curdir, dataset + '_test_short')
    enh_dir = os.path.join(os.curdir, 'restored_' + dataset + '_' + fraction_data)
    
    if not os.path.exists(enh_dir):
        os.mkdir(enh_dir)
    
    filenames = os.listdir(short_dir)
    pathnames = [os.path.join(short_dir, f) for f in filenames]
    
    model_DCNN = torch.load('models/model_DCNN_' + fraction_data + '_' + dataset + '.pth')
    model_DCNN.eval()
    model_DCNN = model_DCNN.cuda(cuda)
    model_CCNN = torch.load('models/model_CCNN_' + fraction_data + '_' + dataset + '.pth')
    model_CCNN.eval()
    model_CCNN = model_CCNN.cuda(cuda)
    
    for path in pathnames:
        
        img_name = os.path.basename(path)
        
        im_short = Image.open(path)
        im_short = np.array(im_short, dtype=np.float32)[:,:,0:3]
        im_short = im_short/255
        im_short = np.moveaxis(im_short, (0,1,2),(1,2,0))
       
        im_short = torch.tensor(im_short, device=cuda)
        im_short = Variable(im_short.unsqueeze(dim=0))
    
        with torch.no_grad():
            im_denoised = model_DCNN(im_short)
    
        if dataset ==  'LOL':
            _, lp_denoised, _ = BuildGPyr(im_denoised)
        else:
            _, _, lp_denoised = BuildGPyr(im_denoised)
    
        with torch.no_grad():
            coefs_est = model_CCNN(lp_denoised)
    
        im_restored = Variable(torch.zeros(im_denoised.size(), dtype = torch.float32, device=cuda))
         
        coefs_est = coefs_est.view(1,3,37,1)
        powers = torch.arange(0.3,1.02,0.02)
        powers = torch.cat((powers, torch.tensor([0], dtype=torch.float32)), dim=0)
        powers = powers.cuda(cuda)
    
        for j in range(len(powers)):
            power = powers[j]
            im_restored = im_restored + (coefs_est[:,:,j:j+1,:]*(im_denoised.pow(power)))
                     
        im_restored = Variable(im_restored[0,:,:,:], requires_grad=False).cpu().numpy()   
        im_restored = np.moveaxis(im_restored, (0,1,2),(2,0,1))
         
        im_restored[im_restored>1] = 1
        im_restored[im_restored<0] = 0
        
        mp.image.imsave(enh_dir+'/'+img_name, im_restored)
        
if __name__=="__main__":
    
    dataset = 'LOL'
    # dataset = 'FUJI'
    # dataset = 'LOL'
    
    # fraction_data = '10%'
    fraction_data = '10%_augmented'
    # fraction_data = '100%'
    
    test_LLIMENet(dataset=dataset, fraction_data=fraction_data)        