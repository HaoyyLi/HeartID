function loc1 = peakdetact(x, fs, dispOn)
    [peaks1,loc1]=findpeaks(x,'MinPeakDistance',fs*60/100);
    [peaks2,loc2]=findpeaks(-x,'MinPeakDistance',fs*60/100);
    
    
    
    
   
    if dispOn==2
        figure
        plot(x)
        hold on
        scatter(loc1,peaks1,'^')
        scatter(loc2,-peaks2,'^')
%         plot([0 max(loc1)],1.5*[mean(peaks1) mean(peaks1)],'r')
%         plot([0 max(loc2)],-1.5*[mean(peaks2) mean(peaks2)],'r')
    end
    % 剪枝 1——上下峰值对应法
    width_thr = 2000;
    for i=1:length(loc1)
        temp=loc1(i,1);
        lc = loc2(find(loc2>temp-width_thr & loc2<temp+width_thr));
        if lc
            loc1(i,2)=lc;
        else
            loc1(i,2)=nan;
        end
    end
    loc1(isnan(loc1(:,2)),:)=[];
    if dispOn==2
        figure
        plot(x)
        hold on
        scatter(loc1(:,1),x(loc1(:,1)),'^')
        scatter(loc1(:,2),x(loc1(:,2)),'^')
%         plot([0 max(loc1(:,1))],1.5*[mean(x(loc1(:,1))) mean(x(loc1(:,1)))],'r')
%         plot([0 max(loc1(:,2))],1.5*[mean(x(loc1(:,2))) mean(x(loc1(:,2)))],'r')
    end
    % 剪枝 2——幅度阈值法
    height_thr_scale = 1.6;
    idx = find(x(loc1(:,1))>height_thr_scale*mean(x(loc1(:,1))) & x(loc1(:,2))<height_thr_scale*mean(x(loc1(:,2))));
    loc1(idx,:)=[];
    if dispOn~=0
        figure
        plot(x)
        hold on
        scatter(loc1(:,1),x(loc1(:,1)),'^')
        scatter(loc1(:,2),x(loc1(:,2)),'^')
        plot([0 max(loc1(:,1))],height_thr_scale*[mean(x(loc1(:,1))) mean(x(loc1(:,1)))],'r')
        plot([0 max(loc1(:,2))],height_thr_scale*[mean(x(loc1(:,2))) mean(x(loc1(:,2)))],'r')
    end
    
%     loc1 = [randn(length(loc2),1), loc2];
    
end