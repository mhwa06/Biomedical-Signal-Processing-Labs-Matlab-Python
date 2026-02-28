%% AF Lab Master (DSAI 2026) - Atrial Fibrillation
% Tasks covered:
% 1) 12-lead visualization (Rva and Ra)
% 2) Simple R-peak detection
% 3) QRST averaging
% 4) Ventricular subtraction (SVD-based)
% 5) Compare with mean QRST subtraction (no SVD)
% 6) Baseline ML for AF recurrence classification on Ra using V1..V6 (7:12)
%
% Files expected in current folder:
% Rva1.mat Rva2.mat Rva3.mat
% Ra1.mat  Ra2.mat  Ra3.mat
% indrecur.mat indnonrecur.mat

clear; clc; close all;

%% -----------------------------
% Load + Merge tensors (75 subjects)
%% -----------------------------
fs = 256;

A = load('Rva1.mat'); B = load('Rva2.mat'); C = load('Rva3.mat');
D = load('Ra1.mat');  E = load('Ra2.mat');  F = load('Ra3.mat');

% Variable names in your files are: Rva1/Rva2/Rva3 and Ra1/Ra2/Ra3
Xva = cat(3, A.Rva1, B.Rva2, C.Rva3);   % 12 x 15000 x 75
Xa  = cat(3, D.Ra1,  E.Ra2,  F.Ra3);    % 12 x 15000 x 75

fprintf('Loaded Xva size: [%d %d %d]\n', size(Xva,1), size(Xva,2), size(Xva,3));
fprintf('Loaded Xa  size: [%d %d %d]\n', size(Xa,1),  size(Xa,2),  size(Xa,3));

leadNames = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
t = (0:size(Xva,2)-1)/fs;

%% -----------------------------
% Choose subject + lead for Q2-Q5
%% -----------------------------
subj = 1;       % 1..75
leadR = 1;      % use lead I as in the handout (you can try lead 2 for better R peaks)

x = squeeze(Xva(leadR,:,subj));

%% ==========================================================
% 1) Visualization of 12-lead ECG (Rva)
%% ==========================================================
figure('Name','Task 1: 12-lead Raw ECG (Xva)','NumberTitle','off');
for l = 1:12
    subplot(12,1,l);
    plot(t, squeeze(Xva(l,:,subj)));
    ylabel(leadNames{l});
    if l == 1, title(['Subject ', num2str(subj), ' - Raw ECG (Xva)']); end
    if l < 12, set(gca,'XTickLabel',[]); else, xlabel('Time (s)'); end
end

% Also show atrial-only (Xa) for the same subject (optional but useful)
figure('Name','Task 1: 12-lead Atrial-only ECG (Xa)','NumberTitle','off');
for l = 1:12
    subplot(12,1,l);
    plot(t, squeeze(Xa(l,:,subj)));
    ylabel(leadNames{l});
    if l == 1, title(['Subject ', num2str(subj), ' - Atrial-only (Xa)']); end
    if l < 12, set(gca,'XTickLabel',[]); else, xlabel('Time (s)'); end
end

%% ==========================================================
% 2) Simple detection of R waves (findpeaks)
%% ==========================================================
% Handout uses: MinPeakHeight = max(x)/2
% Add MinPeakDistance to avoid double-detection (recommended)
minDist = round(0.25*fs); % 250 ms

[PKS, LOCS] = findpeaks(x, ...
    'MinPeakHeight', max(x)/2, ...
    'MinPeakDistance', minDist);

figure('Name','Task 2: R-peaks','NumberTitle','off');
plot(x); hold on;
plot(LOCS, PKS, 'ro');
title(sprintf('R-peak detection | Subject %d | Lead %s', subj, leadNames{leadR}));
xlabel('Samples'); ylabel('Amplitude');
hold off;

fprintf('Detected %d peaks.\n', length(LOCS));

%% ==========================================================
% 3) QRST averaging (build segment matrix R)
%% ==========================================================
% Match the handout logic: segment length uses min(diff(LOCS))
% Segment: LOCS(k)-20 : LOCS(k)+minRR-21  -> length = minRR samples
pre = 20;

if length(LOCS) < 3
    error('Not enough R peaks detected to build QRST segments. Try another lead (e.g., leadR=2) or adjust threshold.');
end

minRR = min(diff(LOCS));          % in samples
segLen = minRR;                   % as implied by handout indexing
startOffset = pre;
endOffset = minRR - pre - 1;      % so that total length becomes minRR

% Keep only peaks where segment is inside bounds
valid = (LOCS - startOffset >= 1) & (LOCS + endOffset <= length(x));
LOCSv = LOCS(valid);

% Build R using k=2..end-1 as handout
if length(LOCSv) < 3
    error('After boundary filtering, not enough peaks. Reduce pre or pick another lead.');
end

R = zeros(length(LOCSv)-2, segLen);
for k = 2:length(LOCSv)-1
    idx0 = LOCSv(k);
    R(k-1,:) = x(idx0-startOffset : idx0+endOffset);
end

figure('Name','Task 3: QRST segments (R)','NumberTitle','off');
plot(R'); title('Aligned QRST segments'); xlabel('Samples in segment'); ylabel('Amplitude');

figure('Name','Task 3: Mean QRST','NumberTitle','off');
plot(mean(R,1),'r','LineWidth',2);
title('Mean QRST template'); xlabel('Samples in segment'); ylabel('Amplitude');

%% ==========================================================
% 4) Ventricular activity subtraction : SVD-based
%% ==========================================================
[U,S,V] = svd(R, 'econ');

figure('Name','Task 4: Singular values','NumberTitle','off');
stem(diag(S));
title('Singular values of R'); xlabel('Component'); ylabel('Singular value');

% Use first 2 components as in handout
M = [V(:,1) V(:,2)];

% Ventricular subtraction on FIRST segment (k=1 in R)
r1 = R(1,:)';
a = pinv(M) * r1;
svd_residual = r1 - M*a;

figure('Name','Task 4: SVD subtraction on first segment','NumberTitle','off');
plot(r1,'LineWidth',1.2); hold on;
plot(M*a,'LineWidth',1.2);
plot(svd_residual,'LineWidth',1.6);
legend('Original segment','Estimated ventricular (SVD)','Residual');
title('SVD-based ventricular subtraction (one segment)');
xlabel('Samples in segment'); ylabel('Amplitude');
hold off;

% Segment Xa at same locations for comparison (handout style)
xa = squeeze(Xa(leadR,:,subj));
RaSeg = zeros(size(R,1), segLen);
for k = 2:length(LOCSv)-1
    idx0 = LOCSv(k);
    RaSeg(k-1,:) = xa(idx0-startOffset : idx0+endOffset);
end

figure('Name','Task 4: Compare ground-truth Ra vs SVD residual','NumberTitle','off');
plot(RaSeg(1,:)', 'LineWidth', 1.6); hold on;
plot(svd_residual, 'LineWidth', 1.6);
legend('Ground truth segment (Xa/Ra)','SVD residual from Xva');
title('Comparison: atrial-only vs SVD residual (same segment)');
xlabel('Samples in segment'); ylabel('Amplitude');
hold off;

%% ==========================================================
% 5) Compare with simple mean QRST subtraction (No SVD)
%% ==========================================================
qrst_mean = mean(R,1)';            % template
mean_residual = r1 - qrst_mean;

figure('Name','Task 5: Mean subtraction vs SVD subtraction','NumberTitle','off');
plot(RaSeg(1,:)', 'LineWidth', 1.6); hold on;
plot(mean_residual, 'LineWidth', 1.6);
plot(svd_residual,  'LineWidth', 1.6);
legend('Ground truth (Xa/Ra)','Mean subtraction residual','SVD subtraction residual');
title('Mean QRST subtraction vs SVD subtraction');
xlabel('Samples in segment'); ylabel('Amplitude');
hold off;

%% ==========================================================
% 6) ML for AF recurrence classification (on Ra), V1..V6 only
%% ==========================================================
% Uses indrecur and indnonrecur (only 63 labeled)
G = load('indrecur.mat');
H = load('indnonrecur.mat');

indrecur = double(G.indrecur(:));
indnonrecur = double(H.indnonrecur(:));

% Safety: remove zeros if any (sometimes stored as uint8 with zeros)
indrecur = indrecur(indrecur >= 1 & indrecur <= size(Xa,3));
indnonrecur = indnonrecur(indnonrecur >= 1 & indnonrecur <= size(Xa,3));

idx = [indrecur; indnonrecur];
y   = [ones(length(indrecur),1); zeros(length(indnonrecur),1)]; % 1=recur, 0=nonrecur

useLeads = 7:12; % V1..V6
Xfeat = zeros(length(idx), length(useLeads)*3);

for i = 1:length(idx)
    s = idx(i);
    feats = [];

    for l = useLeads
        sig = squeeze(Xa(l,:,s));
        sig = sig - mean(sig);

        % Welch PSD
        [pxx,f] = pwelch(sig, hamming(512), 256, 1024, fs);

        band = (f>=3 & f<=12);
        pband = pxx(band); fband = f(band);

        % Features: dominant frequency, bandpower, spectral entropy
        [~,m] = max(pband);
        domF = fband(m);
        bp = trapz(fband, pband);

        pnorm = pband / (sum(pband) + eps);
        sent = -sum(pnorm .* log(pnorm + eps));

        feats = [feats, domF, bp, sent];
    end

    Xfeat(i,:) = feats;
end

% Train/test split + SVM baseline
cv = cvpartition(y,'HoldOut',0.2);
Xtr = Xfeat(training(cv),:); ytr = y(training(cv));
Xte = Xfeat(test(cv),:);     yte = y(test(cv));

mdl = fitcsvm(Xtr, ytr, 'KernelFunction','rbf', 'Standardize', true);
yp  = predict(mdl, Xte);

acc = mean(yp==yte);
fprintf('\nTask 6 SVM baseline accuracy (holdout 20%%): %.3f\n', acc);

figure('Name','Task 6: Confusion Matrix','NumberTitle','off');
confusionchart(yte, yp);
title(sprintf('AF Recurrence Classification (SVM) | Acc=%.3f', acc));
