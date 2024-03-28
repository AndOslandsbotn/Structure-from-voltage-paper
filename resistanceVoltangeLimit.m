epsilon = 0.05;

source = [.1, .1];
sink = [.7, .7];
% source = 2.5;
% sink = .5;

sourceRadius = .1;

distance = 'euclidean';
% corr = -.5;

power = 16:16;
X = [];
voltage = cell(1,length(power));
for i = 1:length(power)
    N = 2^power(i);
    if i==1
        X = rand(N,2);
%         X = 1/3*randn(N,2);
%         X(:,1) = X(:,1) - .7*[ones(N/2,1);-ones(N/2,1)];
%         X = 3*rand(N,1);
    else
        Xnew = rand(N/2,2);
%         Xnew = 1/3*randn(N/2,2);
%         Xnew(:,1) = Xnew(:,1) - .7*[ones(N/4,1);-ones(N/4,1)];
%         Xnew = 3*rand(N/2,1);
        X = [X;Xnew];
    end
    
    
    if strncmp(distance, 'cityblock',9)
        IDX = rangesearch(X,X,epsilon,'Distance','cityblock');
    elseif strncmp(distance, 'euclidean',9)    
        IDX = rangesearch(X,X,epsilon);
    elseif strncmp(distance, 'mahalanobis',9)
        IDX = rangesearch(X,X,epsilon,'Distance','mahalanobis','Cov',[1, corr; corr 1]);
    else
        error('Distance Not Specified');
    end
    
    IDXvec = cell2mat(IDX');
    I = zeros(1,length(IDXvec));
    ind = 0;
    for j=1:N
        I(ind+1 : ind+length(IDX{j})) = j;
        ind = ind + length(IDX{j});
    end
    
    K = sparse(I, IDXvec, ones(1,length(IDXvec)), N, N);
    L = sparse(1:N, 1:N, sum(K,2), N, N) - K;
    
    if strncmp(distance, 'cityblock',9)
        X1 = rangesearch(X,source,sourceRadius,'Distance','cityblock');
        X0 = rangesearch(X,sink,sourceRadius,'Distance','cityblock');
    elseif strncmp(distance, 'euclidean',9)
        X1 = rangesearch(X,source,sourceRadius);
        X0 = rangesearch(X,sink,sourceRadius);
    elseif strncmp(distance, 'mahalanobis', 9)
        X1 = rangesearch(X,source,sourceRadius, 'Distance','mahalanobis','Cov',[1, corr; corr 1]);
        X0 = rangesearch(X,sink,sourceRadius, 'Distance','mahalanobis','Cov',[1, corr; corr 1]);
    end
        
    vconstraint = zeros(N,1);
%   vconstraint(X1{1}) = 1/length(X1{1});
%   vconstraint(X0{1}) = -1/length(X0{1});
     
    % Our method
    vconstraint(X1{1}) = 1;
    vconstraint(X0{1}) = 0;
    Knorm = diag(sum(K, 2).^(-1))*K;
    v = vconstraint;
    for iter = 1:1000
        v = Knorm*v;
        v(X1{1}) = 1;
        v(X0{1}) = 0;
    end
    v = Knorm*v;
    
    % Our method with point source
    vconstraint(X1{1}(1)) = length(X1{1}); %1;
    vconstraint(X0{1}(1)) = 0;
    Knorm = diag(sum(K, 2).^(-1))*K;
    v_point_source = vconstraint;
    for iter = 1:1000
        v_point_source = Knorm*v_point_source;
        v_point_source(X1{1}(1)) = 1;
        v_point_source(X0{1}(1)) = 0;
    end
    v_point_source = Knorm*v_point_source;
    
%     v = (L+sparse(1:N,1:N,1e-12*ones(N,1),N,N)) \ vconstraint;
%     v = v-min(v);
%     v = v*N^2;
%     v = v/max(v);
    %voltage{i} = v;
    
    % Effective resistance region
    vconstraint_er = zeros(N,1);
    vconstraint_er(X1{1}) = 1;
    vconstraint_er(X0{1}) = -1;
    vconstraint_er = vconstraint_er - mean(vconstraint_er);
    
    L = diag(sum(K,2))-K;
    tic
    v_er_region = L\vconstraint_er;
    toc
    
    % Effective resistance region 2
    vconstraint_er2 = zeros(N,1);
    vconstraint_er2(X1{1}(1)) = 1*length(X1{1})^2;
    vconstraint_er2(X0{1}(1)) = -1*length(X0{1})^2;
    vconstraint_er2 = vconstraint_er2 - mean(vconstraint_er2);
    
    L = diag(sum(K,2))-K;
    v_er_region2 = L\vconstraint_er2;
    
    
    % Effective resistance
    vconstraint_er = zeros(N,1);
    vconstraint_er(X1{1}(1)) = 1;
    vconstraint_er(X0{1}(1)) = -1;
    
    L = diag(sum(K,2))-K;
    v_er = L\vconstraint_er;
    
    figure
    if size(X,2)==2
        subplot(1,5,1)
        scatter(X(:,1),X(:,2),20,v,'filled');
        colorbar;
        axis image
        subplot(1,5,2)
        scatter(X(:,1),X(:,2),20,v_point_source,'filled');
        colorbar;
        axis image
        subplot(1,5,3)
        scatter(X(:,1),X(:,2),20,v_er_region,'filled');
        colorbar;
        axis image
        subplot(1,5,4)
        scatter(X(:,1),X(:,2),20,v_er_region2,'filled');
        colorbar;
        axis image
        subplot(1,5,5)
        scatter(X(:,1),X(:,2),20,v_er,'filled');
        colorbar;
        axis image
    else
        [val,ind]=sort(X);
        plot(val,v(ind),'LineWidth',2);
    end
    axis image
    %axis([0,3,-.05, 1.05]);
    %saveas(gcf,['resistanceImages/voltage2Diterative_' num2str(i)],'pdf')
    
end
    