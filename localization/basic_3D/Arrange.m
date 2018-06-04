function [ data ] = Arrange( nr_node, filename )
    
temp_data = load(filename);

data = zeros(nr_node, 1);
for i = 1:nr_node
    count = 0;
    for j = 1:size(temp_data)
        if (temp_data(j, 1) == i+1)
            count = count + 1;
            data(i) = data(i) + temp_data(j, 2);
        end
    end
    data(i)  = data(i)/count; 
end

end
