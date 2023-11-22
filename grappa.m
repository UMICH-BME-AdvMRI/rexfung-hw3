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

% Compute kernel weights
kernelSize = [3 2]; % [Nx Ny]

% Reshape each 3x3 patch in a vector and put all of them into a temp source 
% matrix for each channel of a coil
kNo = 1; % kernel/patch number
for ny = ACS_lines(1):ACS_lines(end)-2
    for nx = 1:(Nx-2)
        for coil = 1:Ncoils
        S_temp(kNo,coil,:) = ...
            reshape(kspace_grappa(nx:nx+2,ny:ny+2,coil)',[1,9]);
        end 
        kNo = kNo + 1; % to move through all patches
    end
end

% Remove three middle ("unknown") values
% The remaiming values form source matrix S, for each channel
S = S_temp(:,:,[1:3,7:9]);
S = reshape(S,size(S,1),size(S,2)*size(S,3));

% Middle points form target vector T for each channel
T = S_temp(:,:,5);

% Invert S to find weights
W = pinv(S) * T;

% Construct source matric from the undersampled image
kNo = 1; % kernel/patch number
for ny = 1:2:(Ny-2)
    for nx = 1:(Nx-2)
        for coil = 1:Ncoils
        S_new_temp(kNo,coil,:) = ...
            reshape(kspace_grappa(nx:nx+2,ny:ny+2,coil)',[1,9]);
        end 
        kNo = kNo + 1; % to move through all patches
    end
end

% Remove three middle ("unknown") values
% The remaiming values form source matrix S, for each channel
S_new = S_new_temp(:,:,[1:3,7:9]);
S_new = reshape(S_new,size(S_new,1),size(S_new,2)*size(S_new,3));

% T_unknown = S_undersampled * W
T_new = S_new * W;

% reshape vector into matrix
T_new_M = reshape(T_new,[Nx-2,Ny/2 - 1,Ncoils]);

% refill kspace
kspace_filled = zeros(size(kspace));
kspace_filled(:,1:2:end,:) = kspace(:,1:2:end,:);
kspace_filled(1:end-2,2:2:end-1,:) = T_new_M;

% display
imgs = ifftshift(ifft2(kspace_filled));
img = sqrt(sum(imgs.^2, 3));
figure;
im(img)