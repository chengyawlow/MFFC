
% Title   : "Multi-Fold Gabor, PCA and ICA Filter Convolution Descriptor for Face Recognition", accepted by IEEE Transactions on Circuits and Systems for Video Technology on 27 September 2017. 
% Authors : Cheng-Yaw Low, Andrew Beng-Jin Teoh, Cong-Jie Ng
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, congjie}@yonsei.ac.kr
% Early Access URL : http://ieeexplore.ieee.org/document/8063938/

% Our implementation is modified from that of PCANet:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ X_HIST ] = MFFC_FeaExtraction( X, V, MFFC )
    
    %% Perform convolutional feature extraction
    [ X ] = MFFC_Conv( X, MFFC.FILT_SZ, MFFC.FILT_NUM, V{ 1 } ); 
        
    %% Spectral Histogram Feature Encoding
    X_HIST = MFFC_SpecHistFeaEncoding( X, MFFC );  
    
    %% Clear ALL
    clearvars -except X_HIST
    
    
    
    