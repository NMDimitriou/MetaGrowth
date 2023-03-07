%% Compute density from spatial distributions of cells
% Author: Nikolaos M. Dimitriou, 
% McGill University, 2020

clear; clc; close all;

tp    = {'D0' 'D2' 'D5' 'D7' 'D9' 'D12' 'D14'};
group = {'A*E*.txt'};
gname = {'AE'};

% Series 1
%group = {'A*C*.txt','A*E*.txt','B*E*.txt','B*N*.txt','B*W*.txt','F*W*.txt'};
%gname = {'AC','AE','BE','BN','BW','FW'};

% Series 2

%group = {'A*C*.txt','A*N*.txt','A*S*.txt','A*W*.txt','A*E*.txt',...
%         'B*C*.txt','B*N*.txt','B*S*.txt','B*W*.txt','B*E*.txt',...
%         'C*C*.txt','C*N*.txt','C*S*.txt','C*W*.txt','C*E*.txt',...
%         'D*C*.txt','D*N*.txt','D*S*.txt','D*W*.txt','D*E*.txt',...
%                    'E*N*.txt','E*S*.txt','E*W*.txt'           ,...
%                    'F*N*.txt',           'F*W*.txt','F*E*.txt'};
%gname = {'Pac_0p5_AC','Pac_0p5_AN','Pac_0p5_AS','Pac_0p5_AW','Pac_0p5_AE',...
%         'Pac_0p05_BC','Pac_0p05_BN','Pac_0p05_BS','Pac_0p05_BW','Pac_0p05_BE',...
%         'Pac_0p005_CC','Pac_0p005_CN','Pac_0p005_CS','Pac_0p005_CW','Pac_0p005_CE',...
%         'Pac_0p0005_DC','Pac_0p0005_DN','Pac_0p0005_DS','Pac_0p0005_DW','Pac_0p0005_DE',...
%              'Control_s2_EN','Control_s2_ES','Control_s2_EW'     ,...
%              'Control_s2_FN',     'Control_s2_FW','Control_s2_FE'};

lg    = length(group);
disp(['Number of datasets: ' num2str(lg)]);
time  = [0 2 5 7 9 12 14];
lt    = length(time);

%run plotopt.m
% For grid points 480x480x176 multiples of 16 (2500x2500x917um space dimensions)
%dsxy = 5.208333333333333; % downscale parameter for xy-plane
%dsz  = 917/176; % downscale parameter for z-dimension
dx=15; % the approximate size of the cell in mu M
sz = ceil([2500/dx,2500/dx,917/dx]); % scale to the cell size

% For the interpolation
x=linspace(0,2.5,sz(1));
z=linspace(0,0.917,sz(3));
[X,Y,Z] = ndgrid(x,x,z);
szq = [480,480,176];
xq=linspace(0,2.5,szq(1));
zq=linspace(0,0.917,szq(3));
[Xq,Yq,Zq] = ndgrid(xq,xq,zq);

%disp(['Initializing density estmation with dsxy=' num2str(dsxy) ', dz=' num2str(dsz)]);
disp(['Initializing density estmation with dxyz=' num2str(dx)]);
%% Import coordinates

disp('Importing coordinates...')
for i=1:lg
    for j=1:lt
       
        samp{i,j}=dir(['res_coord_scaled/' tp{j} '/' group{i}]);
        coord.(gname{i}).(tp{j})=readmatrix([samp{i,j}.folder '/' samp{i,j}.name]);
        count.(gname{i}).(tp{j})=length(coord.(gname{i}).(tp{j})(:,1));
        % Scale them in order for cells to be points
        coord.(gname{i}).(tp{j})(:,1:3)=ceil(coord.(gname{i}).(tp{j})(:,1:3)./dx);
        %coord.(gname{i}).(tp{j})(:,3  )=ceil(coord.(gname{i}).(tp{j})(:,3  )./dsz) ;
        % Shift all coordinates by 1 to remove zeros
        coord.(gname{i}).(tp{j})(:,1:2)=coord.(gname{i}).(tp{j})(:,1:2)+1;

    end
end


%% Convert centroids to density
disp('Converting points to density...')
[X1,X2,X3] = meshgrid(1:szq(1),1:szq(2),1:szq(3));
d          = 3;
grid       = reshape([X1(:),X2(:),X3(:)],szq(1)*szq(2)*szq(3),d);
denscell   = {};
PV = {};
%parpool(lg);
for i=1:lg
    for j=1:lt
        disp(['Calculating density for ' gname{i} ' at day ' tp{j}]);
        cmt = coord.(gname{i}).(tp{j});
        dmt = akde(cmt,grid);
        PV{i,j} = dmt*count.(gname{i}).(tp{j})*(15^3)/(5.21^3);
        %enscell{i,j} = reshape(dmt,size(X1));
        %denscell{i,j} = denscell{i,j}.*(dx^3); % convert PDF to Probability
        %disp(['Interpolating ' gname{i} ' at day ' tp{j}])
        %F = griddedInterpolant(X,Y,Z,denscell{i,j},'linear');
        %PV{i,j} = F(Xq,Yq,Zq);
        disp(['min PV = ' num2str(min(PV{i,j}(:)))])
        %disp(['min Prob = ' num2str(min(denscell{i,j}(:)))])
        disp(['max PV = ' num2str(max(PV{i,j}(:)))])
        %disp(['max Prob = ' num2str(max(denscell{i,j}(:)))])
        %disp(['Plotting interpolated ' gname{i} ' at day ' tp{j}])
        %pvol(PV{i,j},xq,xq,zq,['interp_prob_' gname{i} '_' tp{j}],tp{j},...
        %'Corrected_Density_double_precision/');
    end
end
delete(gcp('nocreate'));


%% Save matrices to binary files

disp('Saving...')
for i=1:lg 
    stat = mkdir('Corrected_Density_double_precision',gname{i}); 
    for j=1:lt
        fileid = fopen(['Corrected_Density_double_precision/' gname{i} '/' ...
	'corr_dens_' gname{i} '_' tp{j} '.bin'],'w'); 
        fwrite(fileid,PV{i,j},'double');
        fclose(fileid);
    end
end

disp('Finished!')
%//////
