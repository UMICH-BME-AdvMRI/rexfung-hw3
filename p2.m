%% BME 599.020 HW 3 Problem 2
% Rex Fung
clear; close all;

load Data_Assignment3_Problem2.mat; % coilmaps, kspaceData
% [Ny Nx Ncoils]

% Change to [Nx Ny Ncoils]
smaps = permute(coilmaps,[2 1 3]); clear coilmaps;
kspace = permute(kspaceData,[2 1 3]); clear kspaceData;

[Nx,Ny,Ncoils] = size(kspace);
%% Part a) Fully sampled image
imgs = zeros(size(kspace));
for coil = 1:Ncoils
    imgs(:,:,coil) = ifftshift(ifft2(kspace(:,:,coil)))./smaps(:,:,coil);
end
img_fs = sqrt(sum(imgs,3));
img_fs(isnan(img_fs)) = 0;

figure;
subplot(121); im(abs(img_fs)); title('Magnitude');
subplot(122); im(angle(img_fs)); title('Phase');
sgtitle('2a. Fully-Sampled Image')

%% Part b) Aliased R=2 image
kspace_R2 = kspace;
kspace_R2(:,2:2:end,:) = 0;

imgs = zeros(size(kspace));
for coil = 1:Ncoils
    imgs(:,:,coil) = ifftshift(ifft2(ifftshift(kspace_R2(:,:,coil)))).*conj(smaps(:,:,coil));
end
img_fs = sum(imgs,3) ./ sum(conj(smaps).*smaps,3);
img_fs(isnan(img_fs)) = 0;

figure;
im(abs(img_fs)); title('Magnitude'); title('2b. 2x Undersampled Image')

%% Part c) SENSE R=2 recon
R = 2;
kspace_R2 = kspace;
kspace_R2(:,2:R:end,:) = 0;
imgs_R2 = ifftshift(ifft2(ifftshift(kspace_R2)));

img = zeros(Nx,Ny);
for nx = 1:Nx
    for ny = Ny/R/2 + 1 : Ny/R*3/2
        % Calculate source pixel locations
        ny1 = ny;
        ny2 = ny + Ny/R;
        if ny2 > Ny
            ny2 = mod(ny2,Ny);
        end

        % Recover pixel values via pseudoinverse
        pixels_aliased = squeeze(imgs_R2(nx,ny,:)); % 8 x 1
        smap_weights = [squeeze(smaps(nx,ny1,:)),...
                        squeeze(smaps(nx,ny2,:))]; % 8 x 2;
        pixels_unaliased = pinv(smap_weights)*pixels_aliased; % 2 x 1
        
        % Allocate recovered pixel values
        img(nx,ny1) = pixels_unaliased(1);
        img(nx,ny2) = pixels_unaliased(2);
    end
end

diff = img - img_fs;

figure;
subplot(121); im(abs(img));
subplot(122); im(abs(diff));
sgtitle('2c. SENSE recon for R = 2. Left is reconstructed image. Right is difference image')

%% Part d) SENSE R=4 recon
R = 4;
kspace_R4 = kspace;
kspace_R4(:,2:R:end,:) = 0;
imgs_R4 = ifftshift(ifft2(ifftshift(kspace_R4)));

img = zeros(Nx,Ny);
for nx = 1:Nx
    for ny = Ny/2 - Ny/R/2 + 1 : Ny/2 + Ny/R/2
        % Calculate source pixel locations
        nys = zeros(1,R);
        for r = 1:R
            loc = ny + (r-1)*Ny/R;
            if loc > Ny
                loc = mod(loc,Ny);
            end
            nys(r) = loc;
        end

        % Recover pixel values via pseudoinverse
        pixels_aliased = squeeze(imgs_R4(nx,ny,:)); % 8 x 1
        smap_weights = zeros(Ncoils,R); % 8 x 4
        for r = 1:R
            smap_weights(:,r) = squeeze(smaps(nx,nys(r),:));
        end
        pixels_unaliased = pinv(smap_weights)*pixels_aliased; % 4 x 1
        
        % Allocate recovered pixel values
        for r = 1:R
            img(nx,nys(r)) = pixels_unaliased(r);
        end
    end
end

diff = img - img_fs;

figure;
subplot(121); im(abs(img));
subplot(122); im(abs(diff));
sgtitle('2d. SENSE recon for R = 4. Left is reconstructed image. Right is difference image')