function preprocess(rootpath)
% 根路径----------该路径下将生成Dataset文件夹保存处理后的数据集
% 原始数据路径
datadir = [rootpath '\原始数据'];
filelist = dir(datadir);
for n=3:length(filelist)
    filename = [datadir '\' filelist(n).name];
    USER_ID = filelist(n).name(1:8);
    savepath = [rootpath '\同步数据\' USER_ID];
    if ~exist(savepath)
        mkdir(savepath)
    end
    tmpdata = load(filename);
    s1 = tmpdata.EarPCG;
    s2 = tmpdata.GT_PCG;
    fs = tmpdata.fs;
    xor = xcorr(s1, s2);
    [~, idx] = max(xor);
    offset = idx - floor(length(xor)/2);
    if(offset>0)
        s1 = s1(offset:end);
    else
        s2 = s2(-offset:end);
    end
    L = min([length(s1),length(s2)]);
    s1 = s1(1:L);
    s2 = s2(1:L);
    EarPCG = s1;
    GT_PCG = s2;
    save([savepath '\' filelist(n).name], 'EarPCG', 'GT_PCG', 'fs');
end
%
datadir = [rootpath '\同步数据'];
savedir = [rootpath '\Dataset'];
if ~exist(savedir)
    mkdir(savedir)
end
userslist = dir(datadir);
for u=3:length(userslist)
    XX = [];
    USER_ID = userslist(u).name;
    savepath = [savedir '\' USER_ID];
    if ~exist(savepath)
        mkdir(savepath)
        mkdir([savepath '/trainset'])
        mkdir([savepath '/validset'])
        mkdir([savepath '/testset'])
    end
    filelist = dir([datadir '\' USER_ID]);
    for n=3:length(filelist)
    filename = [filelist(n).folder '\' filelist(n).name];
    tmpdata = load(filename);
    s1 = tmpdata.EarPCG;
    fs = tmpdata.fs;

     bpFilt = designfilt('lowpassiir', ...
        'PassbandFrequency',45, ...
         'StopbandFrequency',50,...
         'PassbandRipple',0.2,...
         'StopbandAttenuation',65,...
         'DesignMethod','butter','SampleRate',fs);
    s1_bp = filter(bpFilt,s1);

    imf=eemd2(s1_bp,10,100,0.2);
    s1_ewtd=zeros(length(s1),1);
    for i=1:size(imf,2)
        if(estimate_hurst_exponent(imf(:,i)')>0.3)
            temp = wavedenoising(imf(:,i), 8, 'db10', 'heursure', 's', 0);
            s1_ewtd=s1_ewtd+temp;
        end
    end

    loc = peakdetact(s1_ewtd, fs, 0);
    if std(loc(2:end,1)-loc(1:end-1,1))>std(loc(2:end,2)-loc(1:end-1,2))
        idx = loc(:,2);
    else
        idx = loc(:,1);
    end
    idx(find(idx<8000))=[];
    diff = idx(2:end)-idx(1:end-1);
    sig_spc = [0;idx(find(diff>60/55*fs));length(s1_ewtd)];
    sig_spc = [sig_spc(1:end-1)+1 sig_spc(2:end)];
    idx_spc = [0;find(diff>60/55*fs);length(idx)];
    idx_spc = [idx_spc(1:end-1)+1 idx_spc(2:end)];
    split_idx = cell(size(sig_spc,1),2);
    for i=1:size(sig_spc,1)
        split_idx{i,1}=s1_ewtd(sig_spc(i,1):sig_spc(i,2));
        split_idx{i,2}=idx(idx_spc(i,1):idx_spc(i,2))-sig_spc(i,1)+1;
    end
    s1_a = [];
    for i=1:size(split_idx,1)
        tmp = split_idx{i,1};
        pek = [1;split_idx{i,2};length(tmp)];
        sep = pek((1:end-1))+ceil((pek(2:end)-pek(1:end-1))*1/2);
        sep = [1;sep;length(tmp)];
        for j=4:length(sep)
            temp = tmp(sep(j-3):sep(j));
            if length(temp)<15000
                continue
            end
            temp = interp1([1:length(temp)],temp,linspace(1,length(temp),12000),'spline');
            s1_a = [s1_a;temp];
        end
    end
    s1_a(1,:)=[];

    Feature = zeros(67,768,size(s1_a,1));
    for i=1:size(s1_a,1)
        Feature(:,:,i) = abs(cwt(resample(s1_a(i,:),512,8000)));
    end
    XX = cat(3,XX,Feature);
    end
    L = size(XX,3);
    rng(0);
    idx = randperm(L);
    XX = XX(:,:,idx);
    
    L_train = floor(0.6*L);
    L_valid = floor(0.2*L);
    X = XX(:,:,1:L_train);
    for i=1:size(X,3)
        img = squeeze(X(:,:,i));
        figure('visible','off');
        s=surf(img);
        s.LineStyle='none';
        colormap  jet
        view(0,90)
        set(gca,'Xlim',[0, size(img,2)],'Ylim',[0, size(img,1)]);
        set(gca,'CLim',[min(img,[],'all') max(img,[],'all')])
        axis('off');
        set(gca,'position',[0 0 1 1])
        set(gcf,'position',[1920*0.08 1080*0.3 600/1.562 600/1.562]);
        saveas(gcf,[savepath '\trainset\' num2str(i) '.png']);
    end
    X = XX(:,:,L_train+1:L_train+L_valid);
    for i=1:size(X,3)
        img = squeeze(X(:,:,i));
        figure('visible','off');
        s=surf(img);
        s.LineStyle='none';
        colormap  jet
        view(0,90)
        set(gca,'Xlim',[0, size(img,2)],'Ylim',[0, size(img,1)]);
        set(gca,'CLim',[min(img,[],'all') max(img,[],'all')])
        axis('off');
        set(gca,'position',[0 0 1 1])
        set(gcf,'position',[1920*0.08 1080*0.3 600/1.562 600/1.562]);
        saveas(gcf,[savepath '\validset\' num2str(i) '.png']);
    end
    X = XX(:,:,L_train+L_valid+1:end);
    for i=1:size(X,3)
        img = squeeze(X(:,:,i));
        figure('visible','off');
        s=surf(img);
        s.LineStyle='none';
        colormap  jet
        view(0,90)
        set(gca,'Xlim',[0, size(img,2)],'Ylim',[0, size(img,1)]);
        set(gca,'CLim',[min(img,[],'all') max(img,[],'all')])
        axis('off');
        set(gca,'position',[0 0 1 1])
        set(gcf,'position',[1920*0.08 1080*0.3 600/1.562 600/1.562]);
        saveas(gcf,[savepath '\testset\' num2str(i) '.png']);
    end
end