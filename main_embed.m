clc;clear;
%% Parameters Setting
G=9;
Delta=32;
gamma=30;

%% 256bit
MODE = 1; %RRW-PCET
N=20; %
num=256; %
T_init = 8500;%

% MODE = 2; %RRW-PCT
% N=27; %
% num=256; %
% T_init = 6500;

% MODE = 3; %RRW-PST
% N=28; %
% num=256;
% T_init = 8000;

%% Images Reading
file_path =  'image\';
img_path_list = dir(strcat(file_path,'*.bmp'));
img_num = length(img_path_list);

if img_num > 0 
    temp=0;
    for j =1:4
        image_name = img_path_list(j).name;
        image =  imread(strcat(file_path,image_name));
        mysize=size(image);
        if numel(mysize)>2
            if mysize(3) ==2
                image = image(:,:,1);
            else
                image=rgb2gray(image); 
            end
        end
        [image_Rows, image_Cols]=size(image);
        if image_Rows~=512 || image_Cols~=512
            image =imresize(uint8(image),[512,512]);
        end
            temp=temp+1;
            [ psnr1(temp,1) , BER_no_attack(temp,1)]...
                = PHT_version(image, MODE, N, Delta, num, T_init , gamma, G);
            Card(temp,:) = [j;T_init ];
        toc;
    end
end
