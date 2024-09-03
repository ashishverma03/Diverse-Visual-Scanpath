clear ;
parentfolder = 'E:\database\OSIE\data';
image_dir = dir([parentfolder '\stimuli\']);
Resized_fix_dir = dir([parentfolder '\Resized_Traj_image\']);

image_dir(ismember({image_dir.name},{'.','..'})) = [];
Resized_fix_dir(ismember( {Resized_fix_dir.name}, {'.', '..'})) = [];

for img_idx = 1:size(Resized_fix_dir,1)
    
    image_name = [Resized_fix_dir(img_idx).name] ;
    Resized_org_traj = load([ parentfolder '\Resized_Traj_image\' image_name(1:end-4) '.mat']);
    x_cell = Resized_org_traj.Resized_org_traj(cellfun(@(x) ~isequal(x, 0), Resized_org_traj.Resized_org_traj));
    
    num_traj = size(x_cell,2);
    %distance between scanpaths
    dist_bt_traj = zeros(num_traj,num_traj);
    for i = 1:num_traj
        for j = i:num_traj
            dist_bt_traj(i,j) = dtw(x_cell{i}',x_cell{j}',4);
            dist_bt_traj(j,i) = dtw(x_cell{i}',x_cell{j}',4);
        end
    end
    Z = linkage(dist_bt_traj,'complete');
    
    max_clust = 4;
    T = cluster(Z,'maxclust',max_clust);
    cutoff = median([Z(end-3,3) Z(end-2,3) Z(end-1,3)]);
%     dendrogram(Z,'ColorThreshold',cutoff)
    for clust_id = 1: max_clust
        pos_clust{clust_id} = find(T == clust_id)';
        num_clust(clust_id) = length(pos_clust{clust_id});
    end
    
    num_Gen_traj = 25;
    traj_id = 0;
    Label_org_traj = zeros(1,15);
    Gen_traj = cell(2,num_Gen_traj);
    for clust_id = 1:max_clust
        
        % For defining label of original eye trajectories by the degree of
        % agreement between observers (number of observers in corresponding
        % class) in this method of trajectory clustering-------------------
        for clust_no = 1:num_clust(clust_id)
            Label_org_traj(1,pos_clust{clust_id}(clust_no)) = num_clust(clust_id);
        end
        % ---------------------------------------------------------------
        
        if num_clust(clust_id)>2
            
            traj_id1 = traj_id*num_Gen_traj+1;
            traj_id = traj_id+1;
            Traj_cell = cell(1,1);
            for clust_no = 1:num_clust(clust_id)
                Traj_cell{1,clust_no} =  x_cell{1,pos_clust{clust_id}(clust_no)};
            end
            % %-------------------------------------------------
            % % % % HMM modeling for clusters
            updateflags=[1,1,1,1];
            maxiter=50;
            tol=1e-6;
            %T=10
            
            Trans= .2*ones(5,5); p0=.2*ones(5,1);
            C = []; R=[];
            [Trans,p0,C,R,ll] = bw0(Traj_cell,Trans,p0,C,R,tol,maxiter,updateflags);
            
            % Calculating average length of eye trajectories for a given
            % cluster ----------------------------------------------------
            avg_traj_len = 0;
            for l=1:length(Traj_cell)
                avg_traj_len = avg_traj_len + length(Traj_cell{l});
            end
            avg_traj_len =floor(avg_traj_len/length(Traj_cell));
            
            %-------------------------------------------------------------
            
            N = avg_traj_len; % length of trajectories to be generated
            
            %Trajectories generation through HMM model
            for i = traj_id1:num_Gen_traj+traj_id1-1
                [Y,states] = hmm(N,Trans,p0,C,R);
                Gen_traj{1,i} = round(Y);
                Gen_traj{2,i} = num_clust(clust_id);
            end

            dirName = [parentfolder '\HMM_gen_traj_label1\'];
            if ~exist(dirName, 'dir')
                mkdir(dirName);
            end
            cd(dirName);
            filename = image_name(1:end-4);
            save(filename,'Gen_traj');   
        end
        
    end
    
    % % Saving array of labels for Original images according to clustering
    dirName_label = [parentfolder '\Labels_org_traj\'];
    if ~exist(dirName_label, 'dir')
        mkdir(dirName_label);
    end
    cd(dirName_label);
    filename = image_name(1:end-4);
    save([filename '_label'],'Label_org_traj');
    
end
