%% BME 599.020 HW 3 Problem 1
% Rex Fung
clear; close all;

load Data_Assignment3_Problem1.mat; % kspaceData_SingleCoil

kspace = kspaceData_SingleCoil; clear kspaceData_SingleCoil; % [Ny Nx]

% Make y dimension vertical, x dimension horizontal, show reference img
kspace = permute(kspace,[2,1]); % [Nx Ny]
img = ifftshift(ifft2(kspace));
figure;
subplot(121); im('db40',kspace);
subplot(122); im(abs(img));
sgtitle('Ground truth');

% Partial Fourier undersampling
[Ny, Nx] = size(kspace);
pf = 5/8;
kspace_pf = zeros(size(kspace));
kspace_pf(:,1:round(Ny*pf)) = kspace(:,1:round(Ny*pf));

%% Part a) Zero-Filled Recon
img_pf = ifftshift(ifft2(kspace_pf));
figure;
subplot(121); im(abs(img_pf)); title('magnitude');
subplot(122); im(angle(img_pf)); title('phase');
sgtitle('1a. Zero-filled recon');

% Difference img
diff = img - img_pf;
figure;
subplot(121); im(abs(diff)); title('magnitude');
subplot(122); im(angle(diff)); title('phase');
sgtitle('1a. Difference image');

%% Part b) Conjugate Phase Recon
% Low-res phase estimate
kspace_lowres = zeros(size(kspace_pf));
central_lines = ceil(Ny/2 - Ny/16):ceil(Ny/2 + Ny/16) - 1;
hann_filt = repmat(hann(Nx),[1,length(central_lines)]);
kspace_lowres(:,central_lines) = kspace_pf(:,central_lines).*hann_filt;
phase_est = angle(ifftshift(ifft2(kspace_lowres)));

% Iterate
img_pocs = img_pf;
Niters = 1000;
for n = 1:Niters
    img_pocs = abs(img_pocs) .* exp(1i.*phase_est);
    kspace_pocs = fft2(img_pocs);
    kspace_pocs(:,1:Ny*pf) = kspace_pf(:,1:Ny*pf);
    img_pocs = ifft2(kspace_pocs);
end
img_pocs = ifftshift(img_pocs);

% Display
figure;
subplot(121); im(abs(img_pocs)); title('magnitude');
subplot(122); im(angle(img_pocs)); title('phase');
sgtitle('1b. POCS recon');

% Difference img
diff = img - img_pocs;
figure;
subplot(121); im(abs(diff)); title('magnitude');
subplot(122); im(angle(diff)); title('phase');
sgtitle('1b. Difference image');