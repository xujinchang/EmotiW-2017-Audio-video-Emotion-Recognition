function ret = merge_blob(filename)

[filenames] = textread(filename,'%s%*[^\n]');
video_length = length(filenames)
fp1 = fopen('lbp_test_128_result.txt','wt')
for i = 1:video_length
    video_name = char(filenames{i});
    prob_name = dir([video_name '/' '*.prob']);
    tmp =zeros(1,7);
    for j = 1: length(prob_name)
        [s,data]=read_binary_blob([video_name '/' prob_name(j).name]);
        tmp  = tmp+data;
    end
    tmp = tmp / length(prob_name)
    [C,I] = max(tmp);
    fprintf(fp1,'%s \n',[video_name '/' ' ' num2str(I-1)]);
    str = regexp(video_name,'/','split');
    file_name = char(str(end));
    fp = fopen(['./afew_test_prob/' file_name '_prob'],'wt');
    fprintf(fp,'%4f ',tmp);
    fclose(fp);



end

ret = 0;
end

