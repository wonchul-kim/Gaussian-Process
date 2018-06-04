function [packet] = virtual_node(x,node)
a = x(1);
b = x(2);
% angle = x(3);

d0 = 0.5;
P0 = 150;
np = 3;

packet = [];
for i=1:length(node)
    d = sqrt((a-node(i,1))^2 + (b-node(i,2))^2);
    rssi = P0 - 10*np*log(d/d0) + 2*randn(1,1);
    packet = [packet; rssi];
end

