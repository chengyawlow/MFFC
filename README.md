
	Title   : "Multi-Fold Gabor, PCA and ICA Filter Convolution Descriptor for Face Recognition", accepted by IEEE Transactions on Circuits and Systems for Video Technology on 27 September 2017. 
	Authors : Cheng-Yaw Low, Andrew Beng-Jin Teoh, Cong-Jie Ng
	Affl.   : Yonsei University, Seoul, South Korea
	Email   : {chengyawlow, bjteoh, congjie}@yonsei.ac.kr

	This paper appears in: IEEE Transactions on Circuits and Systems for Video Technology 
	Issue Date: <not found>
	Volume: <not found> Issue: <not found>
	On page(s): 0
	Print ISSN: 1051-8215
	Online ISSN: 1558-2205
	Digital Object Identifier: 10.1109/TCSVT.2017.2761829
	Early Access URL : http://ieeexplore.ieee.org/document/8063938/

	Our implementation is modified from that of PCANet:
	T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
	URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html
		
	****************************************

	To derive and evaluate the 2-FFC descriptors:

	1. 	Use only FERET dataset, if you have one; or email us for "FERET_I_128_128.mat".

	2. 	Run FERET_MFFC_Gabor_PCA_ICA_MAIN with the following parameter configurations:
	
		By DEFAULT, MFFC_FLAG = 1 (trigger 2-FFC filter diversification);
		
		FILT_TYPE = 1 : 2-FFC Gabor descriptor.
		
		FILT_TYPE = 2 : 2-FFC Gabor-PCA descriptor.
		
		FILT_TYPE = 3 : 2-FFC Gabor-ICA descriptor.
		
	3.	Refer to the screen shots reposited in the "MFFC_Performance_Summary" folder for performance summary.
		
		
	Other parameter settings include:
	
	1. 	To derive 1-FFC descriptors, set MFFC_FLAG to 0 (LINE 28).
	
	2. 	To trigger max-pooling (instead of mean-pooling) on histogram features, set HistPoolType = 2 (LINE 64);
	
	3. 	To de-activate histogram pooling, set HistPoolRatio = 1 (LINE 71).
	
	
	Note that, the 8 condensed Gabor filters, and the 8 pre-learned PCA and ICA filters (first-layer) are provided in the "MFFC_PRE_LEARNED_FILTERS" folder.
	
	Due to the reason that we pre-learned the PCA and ICA filters from the randomly sampled 500,000 patches (and we have also revised our implementation), you may find some marginal performance discrepancies over the FERET probe sets.
	
	We extend the performance analysis for other 2-FFC descriptor types in:
	C. Y. Low, and A. B. J. Teoh, “Finessing filter scarcity problem in face recognition via multi-fold filter convolution,” in Proc. IWPR, 104430G (2017), DOI: 10.1117/12.2280352.
	
