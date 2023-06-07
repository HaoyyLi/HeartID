function sig=wavedenoising(s,order, type, thrmode, thrmethod, dispOn)

% Input:
%   x: signal
%   order: decomposition level
%   type: worthogonal or biorthogonal wavelet
%   thrmode: 'rigrsure''sqtwolog''heursure''minimaxi'
%   thrmethod: s--soft, h--hard
%   dispOn: figure

[c,l]=wavedec(s,order,type);
approx = appcoef(c,l,type);
cd = detcoef(c,l,[1:order]); 
if dispOn==1
figure
ha(1)=subplot(ceil((order+1)/3),3,1);
plot(approx)
title('Approximation Coefficients')
for i=2:order+1
ha(i)=subplot(ceil((order+1)/3),3,i);
plot(cd{order+2-i})
title(['Level ' num2str(order+2-i) ' Detail Coefficients'])
end
linkaxes(ha, ['y'])
clear ha
end
%     x=cd{1,1};
%     n=length(x);
%     MAD=sum(abs(x-mean(x)))/n;
%     thr = MAD/0.6745*thselect(x,'sqtwolog');
for i=1:length(cd)
    x=cd{1,i};
    n=length(x);
    MAD=sum(abs(x-mean(x)))/n;
    thr = MAD/0.6745*thselect(x,thrmode);        %'rigrsure''sqtwolog''heursure''minimaxi'
    cd{1,i}=wthresh(x,thrmethod,thr);
end
c2=approx;
for i=order:-1:1
    c2=[c2;cd{1,i}];
end
sig=waverec(c2,l,type);
end