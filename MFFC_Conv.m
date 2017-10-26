
% Title   : "Multi-Fold Gabor, PCA and ICA Filter Convolution Descriptor for Face Recognition", accepted by IEEE Transactions on Circuits and Systems for Video Technology on 27 September 2017. 
% Authors : Cheng-Yaw Low, Andrew Beng-Jin Teoh, Cong-Jie Ng
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, congjie}@yonsei.ac.kr
% Early Access URL : http://ieeexplore.ieee.org/document/8063938/

% Our implementation is modified from that of PCANet:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ X_FM ] = MFFC_Conv( X, FILT_SZ, FILT_NUM, V )

    %% Intialize relevant parameters
    [ X_H, X_W ] = size( X{ 1 } );
    
    %% Define ZP_DIM : Zero-padding dimension
    ZP_DIM = ( FILT_SZ - 1 ) / 2;  
    
    %% Zero-pad X, with respect to ZP_DIM
    X_ZP = zeros( X_H + FILT_SZ - 1, X_W + FILT_SZ - 1 );
    X_ZP( ( ZP_DIM + 1 ) : end - ZP_DIM,( ZP_DIM + 1 ) : end - ZP_DIM,: ) = X{ 1 };
    
    %% Extract image patches, and perform patch-mean removal
    X_PATCHES = im2col_mean_removal( X_ZP,[ FILT_SZ, FILT_SZ ] ); 

    %% Convolve X_PATCHES with V, accordingly
    X_FM = cell( FILT_NUM, 1 );
    for FILT_ID = 1 : FILT_NUM   
        X_FM_TEMP = V( :, FILT_ID )' * X_PATCHES;
        X_FM_TEMP = reshape( X_FM_TEMP, X_H, X_W );
        X_FM{ FILT_ID } = X_FM_TEMP;
        clear X_FM_TEMP;           
    end
    
    %% Clear ALL
    clearvars -except X_FM;
    
end

