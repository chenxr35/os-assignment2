y=table2array(addresseslocality);
for i=1:10000
    x(i)=i;
end
x=reshape(x,10000,1);
scatter(x,y);