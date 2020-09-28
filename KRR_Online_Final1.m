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
trainTemp = Temp(28:730);
holdTemp = Temp(736:end-1);

%Reads in our data for the market
holdv = xlsread('hold.xlsx');
trainv = xlsread('train.xlsx');

trainVY=trainv(:,1); %This is the price
trainY=trainVY(29:end); %This is the next days price in our training
for i = 1:length(trainVY)-28
    
    dag1(i)=trainVY(23+i);
    dag2(i)=trainVY(24+i);
    dag3(i)=trainVY(25+i);
    dag4(i)=trainVY(26+i);
    dag5(i)=trainVY(27+i);

end
trainXY=[dag1' dag2' dag3' dag4' dag5']; ;%This is todays price in our training
trainVX=trainv(28:end-1,2:5);
trainX=[trainXY trainVX trainTemp]; 


% Hold Values

holdVY=holdv(:,1);
holdY=holdVY(6:end);
for ii = 1:length(holdVY)-5
    
    hdag1(ii)=holdVY(ii);
    hdag2(ii)=holdVY(ii+1);
    hdag3(ii)=holdVY(ii+2);
    hdag4(ii)=holdVY(ii+3);
    hdag5(ii)=holdVY(ii+4);

end
holdXY=[hdag1' hdag2' hdag3' hdag4' hdag5']; 

holdVX=holdv(5:end-1,2:5);
holdX=[holdXY holdVX holdTemp];

Y = holdY; % This is the correct close values, that we compare to



%% Z-score Normalization 

    x=trainX;
    xh=holdX;
    for i=6:10
    x(:, i) = (x(:, i) - max(x(:, i))) / (max(x(:, i)) - min(x(:, i)));
    xh(:, i) = (xh(:, i) - max(xh(:, i))) / (max(xh(:, i)) - min(xh(:, i)));
    end
  
    trainX=x;
    holdX=xh;

    
   
%% KRR-Online
lambda=0.001;
sigma=0.1;

%n=0.5;
L = length(holdY);
f0 = trainY(end);
F=ones(1,L)*f0;
for ii=2:L
    for i=ii-1:L
        K=exp((-norm(holdX(ii-1,:)-holdX(i,:))^2)/2*sigma^2);
        n=(1/sqrt(ii-1));
        F(ii,i)=F(ii-1,i) +2* n*(holdY(ii-1) - F(ii-1,i))*K;
    end
end

Fd=F(2:end,:);
F_pred=diag(Fd)
F_pred=[f0 F_pred']

figure(3)
plot(Y)
hold on
plot(F_pred, '--')

MAPE = 0;
for i=1:length(Y)
MAPE = MAPE + abs((Y(i)-F_pred(i)) / Y(i))
end

MAPE = MAPE/length(Y)



%% Average percentage movement of the price

%holdY=holdVY(2:end);
%holdXY=ho
perc = 0

for i = 1:length(holdY)
    
    perc = perc + abs( (holdY(i)-holdXY(i))/holdY(i));
    
end

AP = perc/(length(holdY))





