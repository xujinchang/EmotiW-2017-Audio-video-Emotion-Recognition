function ret = extract_lbp(filename)

[filenames] = textread(filename,'%s%*[^\n]');
video_length = length(filenames)
fp1 = fopen('lbp_train.fea','wt');
path = '/home/safe_data_dir/EmotiW2017/AEFW/Train/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points/LBPTOP/';
for i = 1:video_length
    video_name = char(filenames{i});
    str = regexp(video_name,'/','split');
    file_name = char(str(end-1));
    prob_name = [path file_name];
    fea = read_sturct(prob_name);
    fprintf(fp1,'%4f ',fea);
    fprintf(fp1,'\n');
    

end
fclose(fp1);
ret = 0;
end

function [fea] = read_sturct(fea_name)

fea_struct = load(fea_name);
fea = fea_struct.LBPTOPFeat;

end