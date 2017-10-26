
% MODIFIED ON 17 FEBRUARY 2017

function im = im2col_mean_removal( InImg, patchsize12 )

    z = size( InImg, 3 );
    im = cell( z, 1 );

    for i = 1 : z
    
        iim = im2colstep( InImg( :, :, i ), patchsize12 );
        im{i} = bsxfun( @minus, iim, mean(iim) )';
        % im{i} = zscore( iim )';

    end

    im = [ im{ : } ]';
     
    %% Clear all, except im
    clearvars -except im;
    
end

% ORIGINAL
% im = im2col_mean_removal(img,[PatchSize PatchSize]); 
% Zero-pad ith image (in a matrix), and perform patch mean removal

% function im = im2col_mean_removal(varargin)
% 
% NumInput = length(varargin);
% InImg = varargin{1};
% patchsize12 = varargin{2};
% 
% z = size(InImg,3);
% im = cell(z,1);
% 
% if NumInput == 2
%     for i = 1 : z
%         iim = im2colstep(InImg(:,:,i),patchsize12);
%         im{i} = bsxfun(@minus, iim, mean(iim))';
%         % Local Constrast Normalization
%         % Source : http://www.cs.toronto.edu/~ranzato/publications/ranzato_cvpr13.pdf
%         % iim = bsxfun(@minus, iim, mean(iim)); 
%         % im{i} = bsxfun(@minus, iim, mean(iim,2))';
%     end
% else
%     for i = 1:z
%         iim = im2colstep(InImg(:,:,i),patchsize12,varargin{3});
%         im{i} = bsxfun(@minus, iim, mean(iim))'; 
%         % iim = bsxfun(@minus, iim, mean(iim)); 
%         % im{i} = bsxfun(@minus, iim, mean(iim,2))';
%     end 
% end
% 
% im = [im{:}]';
    
    