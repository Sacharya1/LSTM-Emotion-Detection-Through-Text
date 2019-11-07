% % clc; clear all; close all;
% timeDate;
% %% Emotion Classifier
% % This code is submitted for the Final Project of Deep Learning
% 
% url = 'http://nlp.stanford.edu/data/glove.6B.zip';
% 
% if exist('glove.zip', 'file')
%     % File exists.  Do stuff....
% else
%     % File does not exist.
%     fprintf('Downloading 822MB glove data set...');
%     outfilename = websave('glove',url);
%     unzip('glove')
% end
% 
% %% Dictionary and Dataset files preprocessing
% emb = readWordEmbedding("glove.6B.50d.txt");
% rawTexts = readtable('dataset/data.bak.csv');
% 
% charTexts = char(rawTexts.Var1);
% 
% wordsVector = num2cell(zeros(size(charTexts,1),41));
% for n = 1:size(charTexts,1)
%     wordsVector(n,:) = [strsplit(charTexts(n,:)) num2cell(zeros(1,41-size(strsplit(charTexts(n,:)),2)))];
% end
% dataFolded = [wordsVector(:,1:40) num2cell(rawTexts.Var2)];
% % dataFolded = wordsVectorStr(randperm(length(wordsVectorStr)),:);
% 
% Labels=categorical(rawTexts.Var2);
% 
% NumberOfSamples = size(dataFolded,1);
% 
% 
% %% Vectorization
% for i=1:NumberOfSamples
%     for j=1:40
%           Word_in_cell= string(dataFolded(i,j));
%           Word_outside_cell= cell2mat(Word_in_cell);
%           Word_in_Vector= word2vec(emb,Word_outside_cell);
%           VectorArray{j}=Word_in_Vector;
%     end
%           FinalVector{i}=cell2mat(VectorArray')';
% end
% 
% modifiedData= FinalVector;
% modifiedData= reshape(modifiedData,[50,40,1, NumberOfSamples]) ;

for i=1:NumberOfSamples
vec=cell2mat(FinalVector(i));
vecValue=vec.^2;
vector= sum(vecValue);
vector= double(vector);
allVec{i}=vector;
end
dataSet=cell2mat(allVec');
dataSet=normalize(dataSet);
xTrain=dataSet(1:40000,:);
xTest= dataSet(40001:45309,:);
Target= grp2idx(Labels);
yTrain= Target(1:40000,:);
yTest= Target(40001:45309,:);


%% Labeling
% for i=1:NumberOfSamples
%     A=categorical( cell2mat(dataFolded(i,41)));
%     AA(:,i)=A;
% end
% Labels= cell2mat(AA);
% % 
% %% RNN structure
% inputSize = 50;
% numHiddenUnits = 180;
% numClasses = 5;
% layers = [ ...
%     sequenceInputLayer(inputSize)
%     lstmLayer(numHiddenUnits,'OutputMode','last')
%     fullyConnectedLayer(192)
%     reluL
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% 
% options = trainingOptions('sgdm', ...
%     'ExecutionEnvironment','cpu', ...
%     'MaxEpochs',20,...
%     'InitialLearnRate',1e-4, ...
%     'Verbose',true, ...
%     'Plots','training-progress');
% % options = trainingOptions('sgdm', ...
% %     'GradientThreshold',1, ...
% %     'InitialLearnRate',0.01, ...
% %     'Plots','training-progress', ...
% %     'Verbose',0);
% 
% 
% %% Training and testing data partitioning
%  trainingData = modifiedData(1:40000);
% testingData = modifiedData(40001:NumberOfSamples);
% trainingLabels= Labels(1:40000);
% target= Labels(40001:NumberOfSamples);
% 
% % Training
% net = trainNetwork(trainingData, trainingLabels,layers,options);
% 
% %% Save network
% RNNFinal = net;
% save RNNFinal;
% 
% % Testing
% predictedLabel = classify(net,testingData);
% 
% %% Accuracy Calculation
%  [M,~,N]= unique(predictedLabel);
%  [P,~,Q]=unique(target);
% k=0;
% for i=1:5309
%     if (N(i)-Q(i)) ==0
%         k=k+1;
%     end
% end
% 
% Accuracy_in_Percent = (k/5309)*100;
% fprintf("Accuracy in Percent: %d",Accuracy_in_Percent);
% 
% 
% 
% timeDate;
% 
% 
% function timeDate()
%     dateTime = fix(clock);
%     fprintf("\nTime: ");
%     fprintf("%d:%d:%d,  ", dateTime(4),dateTime(5),dateTime(6));
%     fprintf("Date: ");
%     fprintf("%d/%d/%d", dateTime(2),dateTime(3),dateTime(1));
%     fprintf("\n");
% end