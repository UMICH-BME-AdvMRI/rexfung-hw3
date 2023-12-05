%% BME 599.020 HW 3 Problem 2
% Rex Fung
clear; close all;

load Data_Assignment3_Problem1.mat; % kspaceData_SingleCoil
load Data_Assignment3_Problem2.mat; % coilmaps, kspaceData
% [Ny Nx Ncoils]

% Change to [Nx Ny]
kspace_sc = permute(kspaceData_SingleCoil,[2 1]); clear kspaceData_SingleCoil;
smaps = permute(coilmaps,[2 1 3]); clear coilmaps;
kspace = permute(kspaceData,[2 1 3]); clear kspaceData;

[Nx,Ny,Ncoils] = size(kspace);

%% Fully sampled
figure;
im(ifftshift(ifft2(kspace_sc)))

%% 2x undersampled
figure;
im(ifftshift(ifft2(kspace_sc(:,1:2:end))))

%% Zero filled
kspace_zf = kspace_sc;
kspace_zf(:,2:2:end) = 0;
figure;
im(ifftshift(ifft2(kspace_zf)))

%% Zeroth order hold
kspace_zoh = kspace_sc;
kspace_zoh(:,2:2:end) = kspace_sc(:,1:2:end);
figure;
im(ifftshift(ifft2(kspace_zoh)))

%% Linear interpolation
kspace_linint = kspace_sc;
for ny = 2:2:Ny
    if ny == Ny
        kspace_linint(:,ny) = kspace_linint(:,ny-1);
        continue
    end
    kspace_linint(:,ny) = (kspace_linint(:,ny-1) + kspace_linint(:,ny+1))/2;
end
figure;
im(ifftshift(ifft2(kspace_linint)))

%% GRAPPA
kspace_grappa = kspace;
kspace_grappa(:,2:2:end,:) = 0; % 2x undersample

% Fill in ACS region
N_ACS = 24;
ACS_lines = Ny/2 - N_ACS/2 + 1 : Ny/2 + N_ACS/2;
kspace_grappa(:,ACS_lines,:) = kspace(:,ACS_lines,:);

% Compute number of patches in ACS (training) region
Nx_patch = 3; Ny_patch = 3; % 3x3 patch/kernel
x_points = 1:(Nx - (Nx_patch - 1));
y_points = ACS_lines(1):(ACS_lines(end) - (Ny_patch - 1));
Npatches = length(x_points)*length(y_points); % 4356
Nsource_per_patch = Nx_patch*(Ny_patch - 1)*Ncoils; % 48
S_train = zeros(Nsource_per_patch,Npatches); % source pixels
T_train = zeros(1,Npatches); % target pixels

% Collect source and target voxels, then compute weights by pinv
weights = zeros(Ncoils,Nsource_per_patch);
for coil = 1:Ncoils
    n_patch = 1; % keep track of which patch we're on
    for ny = y_points
        for nx = x_points
            voxels_temp = kspace_grappa(nx:nx+2,ny:ny+2,:);
            source_voxels_temp = voxels_temp(:,[1 3],:);
            S_train(:,n_patch) = source_voxels_temp(:); % vectorize
            T_train(n_patch) = voxels_temp(2,2,coil);

            n_patch = n_patch + 1;
        end
    end
    % Invert S to find W according to T = W*S <=> T*pinv(S) = W*S*pinv(S) = W
    % T = target_voxels
    % W = weights
    % S = source voxels
    W = T_train*pinv(S_train);
    weights(coil,:) = W;

    % Check residual
    residual = T_train - W*S_train;
    mean_percent_residual = mean(abs(residual)./abs(T_train)) * 100;
end

%% Construct source matric from the undersampled image
% Now use all of ky
y_points = 1:2:(Ny - (Ny_patch - 1));
Npatches = length(x_points)*length(y_points); % 19503
Nsource_per_patch = Nx_patch*(Ny_patch - 1)*Ncoils; % 48
S_test = zeros(Nsource_per_patch,Npatches); % source pixels

% Collect source and target voxels
kspace_recon = zeros(size(kspace));
for coil = 1:Ncoils
    n_patch = 1; % keep track of which patch we're on
    for ny = y_points
        for nx = x_points
            voxels_temp = kspace_grappa(nx:nx+2,ny:ny+2,:);
            source_voxels_temp = voxels_temp(:,[1 3],:);
            S_test(:,n_patch) = source_voxels_temp(:); % vectorize

            n_patch = n_patch + 1;
        end
    end
    % Compute T
    T_test = weights(coil,:)*S_test;

    % reshape vector into matrix
    T_test = reshape(T_test, [length(x_points), length(y_points)]);

    % refill kspace
    kspace_recon(x_points + 1,y_points + 1,coil) = T_test; % x,y locations offset by 1
end

% Overwrite with acquired data
mask = kspace_grappa ~= 0;
kspace_recon(mask) = kspace_grappa(mask);

%% Compute MSE before and after recon
diff_grappa = kspace - kspace_grappa;
MSE_grappa = mean(abs(diff_grappa),'all')
diff_recon = kspace - kspace_recon;
MSE_recon = mean(abs(diff_recon),'all')

%% Display stuff
close all;

% Before and after GRAPPA, kspace
figure;
subplot(121);
im('db40','row',3,'col',3,kspace_grappa);
title('Before GRAPPA recon');
subplot(122);
im('db40','row',3,'col',3,kspace_recon);
title('Afer GRAPPA recon');

% Before and after GRAPPA, images
imgs_grappa = ifftshift(ifft2(fftshift(kspace_grappa)));
img_grappa = sum(imgs_grappa.*conj(smaps), 3);
imgs_recon = ifftshift(ifft2(fftshift(kspace_recon)));
img_recon = sum(imgs_recon.*conj(smaps), 3);
imgs_truth = ifftshift(ifft2(fftshift(kspace)));
img_truth = sum(imgs_truth.*conj(smaps), 3);

figure;
subplot(231); im(abs(img_grappa)); title('Before GRAPPA recon');
subplot(232); im(abs(img_recon)); title('Afer GRAPPA recon');
subplot(233); im(abs(img_truth)); title('Ground Truth');

[min,max] = bounds(abs(img_grappa) - abs(img_truth),'all');
subplot(234); im(abs(img_grappa) - abs(img_truth),[min max]); %title('Undersampled - Truth');
subplot(235); im(abs(img_recon) - abs(img_truth),[min max]); %title('Recon - Truth');
subplot(236); im(abs(img_grappa) - abs(img_recon),[min max]); %title('Undersampled - Recon');

%% Smaps
figure;
subplot(121); im('row',3,'col',3,smaps);
subplot(122); im('db40','row',3,'col',3,fftshift(fft2(smaps)));