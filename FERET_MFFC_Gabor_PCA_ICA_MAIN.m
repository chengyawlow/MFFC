
% Title   : "Multi-Fold Gabor, PCA and ICA FILT Convolution Descriptor for Face Recognition", accepted by IEEE Transactions on Circuits and Systems for Video Technology on 27 September 2017. 
% Authors : Cheng-Yaw Low, Andrew Beng-Jin Teoh, Cong-Jie Ng
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, congjie}@yonsei.ac.kr
% Early Access URL : http://ieeexplore.ieee.org/document/8063938/

% Our implementation is modified from that of PCANet:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function FERET_MFFC_Gabor_PCA_ICA_MAIN( FILT_TYPE )

    clc;    
    fprintf( '\n' );
    fprintf( ' -----------------------------------------------------------\n' );
    fprintf( '               FERET_MFFC_Gabor_PCA_ICA_MAIN                \n' );
    fprintf( ' -----------------------------------------------------------\n' );    
        
    %% Locate MFFC Utilities Folders
    addpath( './MFFC_FILT' );
    addpath( './MFFC_UTIL' );
    
    %% Load FERET_I_128_128.mat 
    load( 'FERET_I_128_128.mat' );  
            
    %% Initialize MFFC paramters       
    MFFC_FLAG = 1;
    
    FILT_SZ = 7;
    FILT_NUM = 8;
    NUM_MFFC_FOLD = 1;  
    if MFFC_FLAG == 1
        NUM_MFFC_FOLD = 2; 
    end
    FILT_SZ = ones( 1, NUM_MFFC_FOLD ) * FILT_SZ;
    FILT_NUM = ones( 1, NUM_MFFC_FOLD ) * FILT_NUM;
    
    % Define MFFC_FILT_DESCR, with respect to FILT_TYPE
    if FILT_TYPE == 1
        MFFC_FILT_DESCR = 'GABOR';
    elseif FILT_TYPE == 2
        MFFC_FILT_DESCR = 'PCA';
    elseif FILT_TYPE == 3
        MFFC_FILT_DESCR = 'ICA';        
    end
        
    % Define MFFC_DESCR, with respect to MFFC_FLAG
    if MFFC_FLAG == 0
        MFFC_DESCR = [ 'SINGLE-LAYER, 1-FFC ', MFFC_FILT_DESCR, ' NETWORKS' ];
        MFFC_FEA_DESCR = [ '1-FFC ', MFFC_FILT_DESCR ];
    elseif MFFC_FLAG == 1
        MFFC_DESCR = [ 'SINGLE-LAYER, TWO-LAYER FLATTENED 2-FFC GABOR-', MFFC_FILT_DESCR, ' NETWORKS' ];
        MFFC_FEA_DESCR = [ '2-FFC GABOR-', MFFC_FILT_DESCR ];
    end
            
    MFFC.DESCR = MFFC_DESCR;
    MFFC.NUM_FOLD = NUM_MFFC_FOLD;
    MFFC.FILT_SZ = FILT_SZ;
    MFFC.FILT_NUM = FILT_NUM;   
    
    % -----------------------------------------------
    % Set Spectral Histogram Feature Encoding Parameters
    HistPoolType = 1;
    if HistPoolType == 1
        HistPoolDescr = 'Mean-Pool';
    elseif HistPoolType == 2
        HistPoolDescr = 'Max-Pool';
    end
    
    HistPoolRatio = 1;
    HistFeaRatio = 2;
    if MFFC_FLAG == 0 && ( FILT_TYPE == 2 || FILT_TYPE == 3 )
        HistPoolRatio = 1;
        HistFeaRatio = 1;
    end
        
    HistNum = 1;
    if NUM_MFFC_FOLD > 1
        HistNum = MFFC.FILT_NUM( NUM_MFFC_FOLD - 1 );
    end
    
    MFFC.HistNum = HistNum;
    MFFC.HistNumFeaMap = MFFC.FILT_NUM( NUM_MFFC_FOLD );
    MFFC.HistBlkOverlapRatio = 0;
    MFFC.HistImgSz_PX = cat( 2, fa.h, fa.w );
    MFFC.HistBlkSz_BLK = [ 8, 8 ];
    MFFC.HistBlkSz_PX = floor( MFFC.HistImgSz_PX ./ MFFC.HistBlkSz_BLK );
    MFFC.HistPoolType = HistPoolType;
    MFFC.HistPoolDescr = HistPoolDescr;
    MFFC.HistPoolRatio = HistPoolRatio;
    
    HistDim = MFFC.HistNum * prod( MFFC.HistBlkSz_BLK ) * 2 .^ MFFC.HistNumFeaMap * HistFeaRatio;
    MFFC.HistDim = HistDim ./ MFFC.HistPoolRatio;
    
    k_wPCA = 1000;
    MFFC.k_wPCA = k_wPCA;
                   
    %% Load Gabor (condensed), PCA, and ICA FILTs, accordingly
    % Load Gabor FILT sets, i.e., V and W, for each real and imaginary part, respectively   
    load( [ 'CondensedGabor_FS', num2str( FILT_SZ( end ) ), '_NF', num2str( FILT_NUM( end ) ), '.mat' ], 'V_Gabor', 'W_Gabor' );
    V = V_Gabor;
    W = W_Gabor;
               
    % Load pre-learned first-layer PCA FILTs
    if FILT_TYPE == 2
        load( [ 'FERET_PCA_FS', num2str( FILT_SZ( end ) ), '_NF', num2str( FILT_NUM( end ) ), '.mat' ], 'V_PCA' );
        V{2} = normc( V_PCA );
        W{2} = normc( V_PCA );
    % Load pre-learned first-layer PCA FILTs
    elseif FILT_TYPE == 3
        load( [ 'FERET_ICA_FS', num2str( FILT_SZ( end ) ), '_NF', num2str( FILT_NUM( end ) ), '.mat' ], 'V_ICA' );
        V{2} = normc(V_ICA);
        W{2} = normc(V_ICA);       
    end
    
    %% Trigger 2-FFC, with respect to MFFC_FLAG and FILT_TYPE
    if MFFC_FLAG == 1
                       
        % Trigger 2-FFC 
        [ V_2FFC_TEMP, W_2FFC_TEMP, OFF_2FFC_FS, OFF_2FFC_SUM ] = Perform_2FFC( V, W, MFFC );
        
        clear V W;
        
        V{1} = V_2FFC_TEMP;
        W{1} = W_2FFC_TEMP;
            
        % Revise MFFC parameters, accordingly
        MFFC.FILT_SZ = OFF_2FFC_FS;
        MFFC.FILT_NUM = OFF_2FFC_SUM;   
        
        clear V_2FFC_TEMP W_2FFC_TEMP;
               
    elseif MFFC_FLAG == 0
        
        % For 1-FFC Gabor 
        if FILT_TYPE == 1
            V{2} = [];
            W{2} = [];
        
        % For 1-FFC PCA and ICA 
        elseif FILT_TYPE == 2 || FILT_TYPE == 3
            V{1} = V{2};
            V{2} = [];
            clear W;
        end
        
    end

    %% Display MFFC Parameters
    MFFC
          
    %% Extract PCANet features from FERET - FA Images 
    fprintf( '\n' );
    fprintf( '**********' );
    fprintf( '\n' );
    
    fprintf( '\n' );
    fprintf( [ 'EXTRACTING ',  MFFC_FEA_DESCR, ' FEATURES FROM FERET - FA IMAGES ... ' ] );
    fprintf('\n');   
        
    X_TR = double( fa.X ) ./ 255;
    Y_TR = fa.y;
    X_TR_H = fa.h;
    X_TR_W = fa.w;
    
    % Convert columnar X_TR into cells 
    X_TR = mat2imgcell( X_TR, X_TR_H, X_TR_W, 'gray' );
    
    % Initialize X_TR_MFFC, i.e., 2-FFC features
    X_TR_MFFC = zeros( MFFC.HistDim, numel( X_TR ) );
          
    % Extreact 2-FFC features, with respect to IMG_ID 
    for IMG_ID = 1 : numel( X_TR )
               
        if IMG_ID == 1 || mod( IMG_ID, 100 ) == 0 || IMG_ID == numel( X_TR )
            fprintf( '\n' );
            fprintf( 'PROCESSING IMG ID : %d', IMG_ID );
            fprintf( '\n' );
        end
          
        [ X_TR_MFFC_REAL_TEMP ] = MFFC_FeaExtraction( X_TR( IMG_ID ), V, MFFC ); 
                                          
        if MFFC_FLAG == 1 || ( MFFC_FLAG == 0 && FILT_TYPE == 1 )
                
            [ X_TR_MFFC_IMAG_TEMP ] = MFFC_FeaExtraction( X_TR( IMG_ID ), W, MFFC );
               
            % Perform histogram feature pooling, either mean or max, with respect to MFFC.HistPoolType 
            X_TR_MFFC_TEMP = Perform_HistPoolMeanMax( X_TR_MFFC_REAL_TEMP, X_TR_MFFC_IMAG_TEMP, MFFC.HistPoolType, MFFC.HistPoolRatio );
            % X_TR_MFFC_TEMP = Perform_HistPoolMeanMax( sqrt( X_TR_MFFC_REAL_TEMP ), sqrt( X_TR_MFFC_IMAG_TEMP ), MFFC.HistPoolType, MFFC.HistPoolRatio );
                        
        else
                
            X_TR_MFFC_TEMP = X_TR_MFFC_REAL_TEMP;
            % X_TR_MFFC_TEMP = sqrt( X_TR_MFFC_REAL_TEMP );
            
        end
            
        X_TR_MFFC( :, IMG_ID ) = normc( sqrt( X_TR_MFFC_TEMP ) );
        % X_TR_MFFC( :, IMG_ID ) = normc( X_TR_MFFC_TEMP );
                       
        clear X_TR_MFFC_REAL_TEMP X_TR_MFFC_IMAG_TEMP X_TR_MFFC_TEMP;
                                        
    end     
               
    Y_TR_MFFC = Y_TR;
        
    % Display X_TR_MFFC_SZ
    X_TR_MFFC_SZ = size( X_TR_MFFC )
    
    pause( 0.0001 );
                 
    %% Learn wPCA proj. mat. from X_TR_MFFC
    fprintf( '\n' );
    fprintf( 'LEARNING wPCA PROJ. MAT. FROM FERET - FA ... \n' );
    fprintf( '\n' );  
 
    [ X_TR_MFFC_MEAN, eigen_wPCA, ~ ] = wPCA_FAST_GENERALIZED( X_TR_MFFC, k_wPCA, 1 );  
              
    X_TR_MFFC_wPCA = bsxfun(@minus, X_TR_MFFC, X_TR_MFFC_MEAN);
    X_TR_MFFC_wPCA = eigen_wPCA * X_TR_MFFC_wPCA;   
    
    pause( 0.0001 );

    %% Extract 2-FFC features from FERET FB, FC, DUP I, DUP II probe sets
    X_TT_DESCR = { 'FB', 'FC', 'DUP I', 'DUP II' };
    X_TT_ALL = { fb, fc, dup1, dup2 };
    
    % Initialize recogRate
    recogRate = zeros( 1, numel( X_TT_ALL ) );

    for X_TT_ID = 1 : numel( X_TT_ALL )
                
        fprintf( '\n' );
        fprintf( '**********' );
        fprintf( '\n' );
        
        fprintf( '\n' );
        fprintf( [ 'EXTRACTING ', MFFC_FEA_DESCR, ' FEATURES FROM FERET - ', X_TT_DESCR{ X_TT_ID }, ' IMAGES ... ' ] );
        fprintf( '\n' );
        
        X_TT = double( X_TT_ALL{ X_TT_ID }.X ) ./ 255;
        Y_TT = X_TT_ALL{ X_TT_ID }.y;
        X_TT_H = X_TT_ALL{ X_TT_ID }.h;
        X_TT_W = X_TT_ALL{ X_TT_ID }.w;
        
        % Convert columnar X_TT into cell representation
        X_TT = mat2imgcell( X_TT, X_TT_H, X_TT_W, 'gray' ); 
        
        % Initialize X_TT_MFFC, i.e., 2-FFC features
        X_TT_MFFC = zeros( MFFC.HistDim, numel( X_TT ) );

        % Extract 2-FFC features, with respect to IMG_ID
        for IMG_ID = 1 : numel( X_TT )
            
            if IMG_ID == 1 || mod( IMG_ID, 100 ) == 0 || IMG_ID == numel( X_TT )
                fprintf( '\n' );
                fprintf( 'PROCESSING IMG ID : %d', IMG_ID );
                fprintf( '\n' );
            end
                        
            [ X_TT_MFFC_REAL_TEMP ] = MFFC_FeaExtraction( X_TT(IMG_ID), V, MFFC );
                      
            if MFFC_FLAG == 1 || ( MFFC_FLAG == 0 && FILT_TYPE == 1 )
                
                [ X_TT_MFFC_IMAG_TEMP ] = MFFC_FeaExtraction( X_TT(IMG_ID), W, MFFC );
                
                % Perform histogram feature pooling, either mean or max, with respect to MFFC.HistPoolType 
                X_TT_MFFC_TEMP = Perform_HistPoolMeanMax( X_TT_MFFC_REAL_TEMP, X_TT_MFFC_IMAG_TEMP, MFFC.HistPoolType, MFFC.HistPoolRatio ); 
                % X_TT_MFFC_TEMP = Perform_HistPoolMeanMax( sqrt( X_TT_MFFC_REAL_TEMP ), sqrt( X_TT_MFFC_IMAG_TEMP ), MFFC.HistPoolType, MFFC.HistPoolRatio ); 
                
            
            else
                
                X_TT_MFFC_TEMP = X_TT_MFFC_REAL_TEMP;
                % X_TT_MFFC_TEMP = sqrt( X_TT_MFFC_REAL_TEMP );
            
            end
            
            X_TT_MFFC( :, IMG_ID ) = normc( sqrt( X_TT_MFFC_TEMP ) );
            % X_TT_MFFC( :, IMG_ID ) = normc( X_TT_MFFC_TEMP );
            
            clear X_TT_MFFC_REAL_TEMP X_TT_MFFC_IMAG_TEMP X_TT_MFFC_TEMP;
                                    
        end   

        Y_TT_MFFC = Y_TT;
        
        X_TT_MFFC_SZ = size( X_TT_MFFC )
   
        %% Apply learned wPCA proj. mat. to X_TT_MFFC
        fprintf( '\n' );
        fprintf( [ 'APPLYING wPCA PROJ. MAT. TO FERET - ', X_TT_DESCR{ X_TT_ID }, '... ' ] );
        fprintf( '\n' );   
        
        X_TT_MFFC_wPCA = bsxfun( @minus, X_TT_MFFC, X_TR_MFFC_MEAN );
        X_TT_MFFC_wPCA = eigen_wPCA * X_TT_MFFC_wPCA;

        %% Calculate recogRate_CD, recogRate_SD
        fprintf( '\n' );
        fprintf( [ 'CALCULATING recogRate FOR FERET - ', X_TT_DESCR{ X_TT_ID }, '... ' ] );
        fprintf( '\n' ); 
        
        [ recogRate_TEMP ] = recognitionRate_CosineDistance( X_TR_MFFC_wPCA, X_TT_MFFC_wPCA, Y_TR_MFFC, Y_TT_MFFC )
        
        recogRate( X_TT_ID ) = recogRate_TEMP;
        
        clear recogRate_TEMP;
        pause( 0.0001 );
                
    end
    
    %% Display performance summary   
    fprintf( '\n' );
    fprintf( '***** PERFORMANCE SUMMARY *****' );
    fprintf( '\n' ); 
    
    recogRate = cat( 2, recogRate, mean( recogRate ) )
           
    %% Display MFFC Parameters
    MFFC
    
end
