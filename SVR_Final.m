clc, clear all, close all

%Reading temperature
Temp1 = xlsread('Temp_rain_2015_2017',2);
Temp2 = xlsread('Temp_rain_2015_2017',3);
Temp3 = xlsread('Temp_rain_2015_2017',4);
Temp4 = xlsread('Temp_rain_2015_2017',5);
Temp5 = xlsread('Temp_rain_2015_2017',6);
Temp6 = xlsread('Temp_rain_2015_2017',7);
Temp7 = xlsread('Temp_rain_2015_2017',8);
Temp8 = xlsread('Temp_rain_2015_2017',9);
Temp9 = xlsread('Temp_rain_2015_2017',10);
Temp10 = xlsread('Temp_rain_2015_2017',11);

Temp1 = Temp1(:,1);
Temp2 = Temp2(:,1);
Temp3 = Temp3(:,1);
Temp4 = Temp4(:,1);
Temp5 = Temp5(:,1);
Temp6 = Temp6(:,1);
Temp7 = Temp7(:,1);
Temp8 = Temp8(:,1);
Temp9 = Temp9(:,1);
Temp10 = Temp10(:,1);

Temp=(Temp1+Temp2+Temp3+Temp4+Temp5+Temp6+Temp7+Temp8+Temp2+Temp10)/10;

%% Fixing data into our vectors
%Reds in our data for the market
trainTemp = Temp(26:730);
holdTemp = Temp(734:end-1);
holdv = xlsread('hold.xlsx');
trainv = xlsread('train.xlsx');

trainVY=trainv(:,1); %This is the price
trainY=trainVY(27:end); %This is the next days price in our training
for i = 1:length(trainVY)-26
    
    dag1(i)=trainVY(23+i);
    dag2(i)=trainVY(24+i);
    dag3(i)=trainVY(25+i);

end
trainXY=[dag1' dag2' dag3']; ;%This is todays price in our training
trainVXn=trainv(26:end-1,2:3);
trainVX=trainv(26:end-1,4:5);

trainX=[trainXY trainVX trainVXn trainTemp]; 



% Hold Values

holdVY=holdv(:,1);
holdY=holdVY(4:end);
for ii = 1:length(holdVY)-3
    
    hdag1(ii)=holdVY(ii);
    hdag2(ii)=holdVY(ii+1);
    hdag3(ii)=holdVY(ii+2);

end
holdXY=[hdag1' hdag2' hdag3']; 

holdVXn=holdv(3:end-1,2:3);
holdVX=holdv(3:end-1,4:5);
holdX=[holdXY holdVX holdVXn holdTemp];


Y = holdY; % This is the correct close values, that we compare to
%trainX=[trainTemp];
%holdX=[holdTemp];

%% Z-score Normalization 

    x=trainX;
    xh=holdX;
    for i=4:8
    x(:, i) = (x(:, i) - max(x(:, i))) / (max(x(:, i)) - min(x(:, i)));
    xh(:, i) = (xh(:, i) - max(xh(:, i))) / (max(xh(:, i)) - min(xh(:, i)));
    end

    trainX=x;
    holdX=xh;
   

%% Cross Validation

%Cv=[0.1 0.01 0.001];
sigmav = [1000 100 10 1];
%epsv=[1 0.1];
% C = 0.1  eps = 0.1
C = 0.1;
e = 0.1;
L = length(trainX);
chop = L/5;


for kk=1:length(sigmav)

    sigma = sigmav(kk);
    error=0;
    
    for ii = 1:210:length(trainX)-210
              
        holdXCV=trainX(ii:ii+210,:);
        holdYCV=trainY(ii:ii+210,:);
        trainXCV=trainX;
        trainXCV([ii:ii+210],:)=[];
        trainYCV=trainY;
        trainYCV([ii:ii+210],:)=[];
        
        
        for j=1:length(trainXCV)
        for jj=1:length(trainXCV)
        %Training
        K=exp((-norm(trainXCV(j,:)-trainXCV(jj,:))^2)/(2*sigma^2));
        Kv(j,jj)=K;
        end
        end

le = length(trainXCV);        
%C=Cv(kk);
%e = epsv(kkk);

cvx_begin
    variables w(le)  b s(le) sh(le)
    minimize ( 0.5*norm(w,2) + C*sum(s+sh) ) 
    subject to
        trainYCV - Kv*w - b <= e + s
        Kv*w + b - trainYCV <= e + sh
        s >= 0
        sh >= 0
cvx_end


    for t=1:length(holdXCV)
    for tt=1:length(trainXCV)
    Kt=exp((-norm(trainXCV(tt,:)-holdXCV(t,:))^2) /(2*sigma^2));  %Creates the "kernel" with our data for prediction
    KTV(tt,1)= Kt;
    end
    yhat(t)=w'*KTV + b ;        %Gives our predictions
    end
    
    error(ii)=mean(sqrt( (holdYCV - yhat').^2));
    end
    
errorv(kk)=mean(error);

end




%% SVR -  Kernel 

sigma=1000;
for i=1:length(trainX)
for ii=1:length(trainX)
     %Training
     K=exp( (-norm(trainX(i,:)-trainX(ii,:))^2) /(2*sigma^2));
     Kv(i,ii)=K;
end
end

%From CV
C=0.1; 
e = 0.1;
le=length(trainY);

cvx_begin
    variables w(le)  b s(le) sh(le)
    minimize ( 0.5*norm(w,2) + C*sum(s+sh) ) 
    subject to
        trainY - Kv*w - b <= e + s
        Kv*w + b - trainY <= e + sh
        s >= 0
        sh >= 0
cvx_end


%% Prediction

for i=1:length(holdX)
for ii=1:length(trainX)
    Kt=exp( (-norm(trainX(ii,:)-holdX(i,:))^2) / (2*sigma^2) );  %Creates the "kernel" with our data for prediction
    KTV(ii,1)= Kt;
end
y_primalK(i)=w'*KTV + b ;        %Gives our predictions
end

figure(2)
plot(Y)
hold on
grid on
plot(y_primalK, '--')
legend('Actual Price', 'Predicted Price')
title('Support Vector Regression')
xlabel('Day', 'fontweight', 'bold')
ylabel('Electricity price / SEK', 'fontweight', 'bold')
axis([0 362 100 600])



MAPE = 0;
for i=1:length(Y)
MAPE = MAPE + abs((Y(i)-y_primalK(i)) / Y(i))


end

MAPE = MAPE/length(Y)

        

    