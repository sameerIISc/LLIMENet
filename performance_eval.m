clear all
clc

folders = 'restored_SONY_100%/';
folderl = 'SONY_test_long/';

files = dir(strcat(folders, '*.png'));

for i = 1:length(files)
    i
    name = files(i).name;

    Is = imread(strcat(folders, name));
    long_file = dir(strcat(folderl, name(1:5), '*.png'));
    Il = imread(strcat(folderl, long_file.name));

    pnr(i) = psnr(Il,Is);
    simc(i) = ssim(Il,Is);
end 
mean(simc)
mean(pnr)