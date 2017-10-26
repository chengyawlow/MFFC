
% Title   : "Multi-Fold Gabor, PCA and ICA Filter Convolution Descriptor for Face Recognition", accepted by IEEE Transactions on Circuits and Systems for Video Technology on 27 September 2017. 
% Authors : Cheng-Yaw Low, Andrew Beng-Jin Teoh, Cong-Jie Ng
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, congjie}@yonsei.ac.kr
% Early Access URL : http://ieeexplore.ieee.org/document/8063938/

% Our implementation is modified from that of PCANet:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ X_POOLED ] = Perform_HistPoolMeanMax( X_1, X_2, MEAN_MAX, Q )

    %% Initialize Q, and MEAN_MAX
    if nargin == 2
        Q = 2;
        MEAN_MAX = 1;
    elseif nargin == 3
        MEAN_MAX = 1;
    end
    
    %% Initialize X_POOLED
    X_POOLED = zeros( ( size( X_1, 1 ) +  size( X_2, 1 ) ) ./ Q, size( X_1, 2 ) );
    assert( size( X_1, 2 ) == size( X_2, 2 ) );
       
    %% Perform histogram faeture pooling, with respect to Q and MEAN_MAX
    for ID = 1 : size( X_1, 2 )

        X_POOLED_TEMP = cat( 1, X_1( :, ID ), X_2( :, ID ) );
        X_POOLED_TEMP = reshape( X_POOLED_TEMP, Q, [] );
        
        % Mean-Pooling
        if MEAN_MAX == 1
            X_POOLED( :, ID ) = mean( X_POOLED_TEMP, 1 );
        % Max-Pooling
        elseif MEAN_MAX == 2
            X_POOLED( :, ID ) = max( X_POOLED_TEMP, [], 1 );
        end
        
        clear X_POOLED_TEMP;
        
    end

    %% Clear ALL
    clearvars -except X_POOLED;
    
end
