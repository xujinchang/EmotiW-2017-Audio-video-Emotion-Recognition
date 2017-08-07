function ret = merge_fc6(filename)

[filenames] = textread(filename,'%s%*[^\n]');
video_length = length(filenames)
fp1 = fopen('c3d_val_fc7.fea','wt');
path = '/local/home/share/xujinchang/project/C3D/C3D-v1.0/examples/c3d_feature_extraction/output/afew_face_val/';
for i = 1:video_length
    video_name = char(filenames{i});
    str = regexp(video_name,'/','split');
    file_name = char(str(end));
    video_path = [path file_name];
    fc6_name = dir([video_path '/' '*.fc7-1']);
    for j = 1: length(fc6_name)
        [s,data]=read_binary_blob([video_path '/' fc6_name(j).name]);
        fprintf(fp1,'%4f ',data);
    	fprintf(fp1,'\n');
 
    end

end
fclose(fp1);
ret = 0;
end
