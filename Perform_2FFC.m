
% Title   : "Multi-Fold Gabor, PCA and ICA FILT Convolution Descriptor for Face Recognition", accepted by IEEE Transactions on Circuits and Systems for Video Technology on 27 September 2017. 
% Authors : Cheng-Yaw Low, Andrew Beng-Jin Teoh, Cong-Jie Ng
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, congjie}@yonsei.ac.kr
% Early Access URL : http://ieeexplore.ieee.org/document/8063938/

% Our implementation is modified from that of PCANet:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ V_OFF, W_OFF, OFF_FS, OFF_SUM ] = Perform_2FFC( V, W, MFFC )

    %% Initialize relevant parameters
    V_F1 = V{1};
    V_F2 = V{2};
        
    W_F1 = W{1};
    W_F2 = W{2};
    
    % Ori. filter size, prior to 2FFC 
    ORI_F1_FS = MFFC.FILT_SZ(1);    
    ORI_F2_FS = MFFC.FILT_SZ(2);
    
    % OFF_SUM : 2FFC offspring filter capacity
    OFF_SUM = prod( MFFC.FILT_NUM );
    
    % OFF_FS : 2FFC offspring filter size        
    OFF_FS = sum( MFFC.FILT_SZ ) - 1; 
            
    %% Perform 2-fold FILT-to-FILT convolution for V 
    V_OFF = zeros( OFF_FS * OFF_FS, OFF_SUM );
    V_OFF_ID = 0;
    
    for V_F1_ID = 1 : size( V_F1, 2 ) 
            
        V_F1_TEMP = V_F1( :, V_F1_ID );
        V_F1_TEMP = reshape( V_F1_TEMP, ORI_F1_FS, ORI_F1_FS );
            
        for V_F2_ID = 1 : size( V_F2, 2 )
                
            V_OFF_ID = V_OFF_ID + 1;
              
            V_F2_TEMP = V_F2( :, V_F2_ID );
            V_F2_TEMP = reshape( V_F2_TEMP, ORI_F2_FS, ORI_F2_FS );
               
            V_MFFC_TEMP = conv2( V_F1_TEMP, V_F2_TEMP );
            V_OFF( :, V_OFF_ID ) = normc( V_MFFC_TEMP(:) );

        end      
            
    end
        
    %% Perform 2-fold FILT-to-FILT convolution for W
    W_OFF = zeros( OFF_FS * OFF_FS, OFF_SUM );
    W_OFF_ID = 0;
    
    for W_F1_ID = 1 : size( W_F1, 2 ) 
            
        W_F1_TEMP = W_F1( :, W_F1_ID );
        W_F1_TEMP = reshape( W_F1_TEMP, ORI_F1_FS, ORI_F1_FS );
            
        for W_F2_ID = 1 : size( W_F2, 2 )
                
            W_OFF_ID = W_OFF_ID + 1;
                
            W_F2_TEMP = W_F2( :, W_F2_ID );
            W_F2_TEMP = reshape( W_F2_TEMP, ORI_F1_FS, ORI_F2_FS );
               
            W_OFF_TEMP = conv2( W_F1_TEMP, W_F2_TEMP );
            W_OFF( :, W_OFF_ID ) = normc( W_OFF_TEMP(:) );
                
        end      
            
    end    
        
    %% Validate V_OFF and W_OFF
    assert( numel( V_OFF ) == OFF_FS * OFF_FS * OFF_SUM );
    assert( numel( W_OFF ) == OFF_FS * OFF_FS * OFF_SUM );
    
    %% Clear ALL
    clearvars -except V_OFF W_OFF OFF_SUM OFF_FS;
  
 end
