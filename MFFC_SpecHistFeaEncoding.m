
% Title   : "Multi-Fold Gabor, PCA and ICA Filter Convolution Descriptor for Face Recognition", accepted by IEEE Transactions on Circuits and Systems for Video Technology on 27 September 2017. 
% Authors : Cheng-Yaw Low, Andrew Beng-Jin Teoh, Cong-Jie Ng
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, congjie}@yonsei.ac.kr
% Early Access URL : http://ieeexplore.ieee.org/document/8063938/

% Our implementation is modified from that of PCANet:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ X_HIST ] = MFFC_SpecHistFeaEncoding( X, MFFC )

    %% Initialize Parameters   
    % Set feature encoding weights W for binary to decimal conversion
    W = 2.^( MFFC.HistNumFeaMap - 1 : -1 : 0 );
    
    NUM_HIST = numel( X ) / MFFC.HistNumFeaMap;
    assert( NUM_HIST == MFFC.HistNum );
    
    X_HIST = [];
       
    %% Perform LBP-like feature encoding
    for HIST_ID = 1 : NUM_HIST 
        
        T = 0;
        for HIST_FEA_MAP_ID = 1 : MFFC.HistNumFeaMap
            % T = T + W( HIST_FEA_MAP_ID ) * Heaviside( X( :, :, ( HIST_ID - 1 ) * MFFC.HistNumFeaMap + HIST_FEA_MAP_ID ) );
            T = T + W( HIST_FEA_MAP_ID ) * Heaviside( X{ ( HIST_ID - 1 ) * MFFC.HistNumFeaMap + HIST_FEA_MAP_ID } );
        end
                        
        % Perform LBP-alike histograming for each local block in T
        STR = MFFC.HistBlkSz_PX;
        FEA_MAP_SZ = size( T );
        if MFFC.HistBlkOverlapRatio ~= 0
            STR = round( ( 1 - MFFC.HistBlkOverlapRatio ) * MFFC.HistBlkSz_PX );
        end
                
        % T : COLUMNAR
        T = im2col_general( T, MFFC.HistBlkSz_PX, STR );
                      
        % X_HIST & X_HIST_TEMP : COLUMNAR
        X_HIST_TEMP = histc( T, ( 0 : 2 ^ MFFC.HistNumFeaMap - 1 )' );
        
        % Trim X_HIST_TEMP, with respect to MFFC.HistBlkSz_BLK
        if MFFC.HistBlkOverlapRatio ~= 0  
            X_HIST_TEMP = spp( X_HIST_TEMP, FEA_MAP_SZ, STR, MFFC );
        end
                         
        X_HIST = cat( 2, X_HIST, X_HIST_TEMP );
        clear X_HIST_TEMP;
        
    end
    
    %% Perform Feature Vectorization on X_HIST
    X_HIST = vec( X_HIST );
    
    %% Clear all, except X_HIST
    clearvars -except X_HIST

end

%% Binary Quantization
function X = Heaviside( X ) 
    X = sign( X );
    X( X <= 0 ) = 0;
end

%% Feature Vectorization
function X = vec( X ) 
    X = X(:);
end

%% SPP Encoding
% function beta = spp( blkwise_fea, sam_coordinate, ImgSize, pyramid )
function beta = spp( blkwise_fea, ImgSize, stride, MFFC )

    x_start = ceil( MFFC.HistBlkSz_PX(2) / 2 );
    y_start = ceil( MFFC.HistBlkSz_PX(1) / 2 );
    x_end = floor( ImgSize(2) - MFFC.HistBlkSz_PX(2) / 2 );
    y_end = floor( ImgSize(1) - MFFC.HistBlkSz_PX(1) / 2 );
                
    sam_coordinate = [...
                    kron( x_start : stride : x_end, ones( 1, length( y_start : stride : y_end ) ) ); 
                    kron( ones( 1, length( x_start : stride : x_end ) ), y_start : stride : y_end ) ];

    [dSize, ~] = size(blkwise_fea);

    img_width = ImgSize(2);
    img_height = ImgSize(1);

    % spatial levels
    pyramid = MFFC.SpatialPyramid_LEVEL;
    pyramid_Levels = length(pyramid);
    pyramid_Bins = pyramid.^2;
    tBins = sum(pyramid_Bins);

    beta = zeros(dSize, tBins);
    % beta = [];
    cnt = 0;

    for i1 = 1:pyramid_Levels
    
        Num_Bins = pyramid_Bins(i1);
    
        wUnit = img_width / pyramid(i1);
        hUnit = img_height / pyramid(i1);
    
        % find to which spatial bin each local descriptor belongs
        xBin = ceil(sam_coordinate(1,:) / wUnit);
        yBin = ceil(sam_coordinate(2,:) / hUnit);
        idxBin = (yBin - 1)*pyramid(i1) + xBin;
    
        for i2 = 1 : Num_Bins     
            cnt = cnt + 1;
            sidxBin = find( idxBin == i2 );
            if isempty(sidxBin)
                continue;
            end      
            % beta( :, cnt ) = max( blkwise_fea( :, sidxBin ), [], 2 );
            beta( :, cnt ) = mean( blkwise_fea( :, sidxBin ), 2 );
            % beta( :, cnt ) = sum( blkwise_fea( :, sidxBin ), 2 );
        end
        
    end
    
end

