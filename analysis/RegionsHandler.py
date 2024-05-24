import graphIO
import numpy as np

class RegionsHandler:
    def __init__(self, region_mappings):
        self.region_mappings = region_mappings
        self.attributes={}
        self.lh_regions = {}
        self.rh_regions = {}
        for key, value in self.region_mappings.items():
            region_name = value.replace('lh_', '').replace('rh_', '')
            if 'lh' in value:
                self.lh_regions[region_name] = key-1
            else:
                self.rh_regions[region_name] = key-1
        if len(self.lh_regions) != len(self.rh_regions):
            raise ValueError(
                "The number of regions in the left and right hemispheres must be equal.")
        for key in self.lh_regions.keys():
            if key not in self.rh_regions:
                raise ValueError(
                    f"Region '{key}' is missing from the right hemisphere.")
        for key in self.rh_regions.keys():
            if key not in self.lh_regions:
                raise ValueError(
                    f"Region '{key}' is missing from the left hemisphere.")
        self.region_tuples = []
        for region_name in self.lh_regions.keys():
            lh_index = self.lh_regions[region_name]
            rh_index = self.rh_regions[region_name]
            self.region_tuples.append((region_name, lh_index-1, rh_index-1))
        self.region_names = list(self.lh_regions.keys())

    def size(self):
        return len(self.region_tuples)
    
    def region_index(self, region_name):
        for (name, lh_index, rh_index) in self.region_tuples:
            if name == region_name:
                return (lh_index, rh_index)
        raise ValueError(f"Region '{region_name}' not found.")
    
    def region_name(self, index):
        for (name, lh_index, rh_index) in self.region_tuples:
            if lh_index == index or rh_index == index:
                return name
        raise ValueError(f"Region with index '{index}' not found.")
    
    def attach_adj_matrix(self, adj_matrix):
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("The adjacency matrix must be square.")
        if adj_matrix.shape[0] != self.size()*2:
            raise ValueError(
                "The adjacency matrix must have the same number of rows as the number of regions.")
        self.adj_matrix = adj_matrix

    def save_adj_matrix(self, adj_matrix_path):
        graphIO.write_adj_matrix(adj_matrix_path, self.adj_matrix)
    
    def load_nodes(self, nodes_path):
        self.nodes = graphIO.read_nodes(nodes_path)
        self.region_coords = []
        for node in self.nodes:
            self.region_coords.append((node[0], node[1], node[2]))
        
    def add_attribute(self, attribute_name, attribute_values):
        if len(attribute_values) != self.size()*2 and len(attribute_values) != self.size():
            raise ValueError(
                "The number of attribute values must be equal to the number of regions.")
        if len(attribute_values) == self.size():
            new_attribute_values = [0] * (self.size() * 2)
            for i, region in enumerate(self.region_names):
                new_attribute_values[self.lh_regions[region]] = attribute_values[i]
                new_attribute_values[self.rh_regions[region]] = attribute_values[i]
            attribute_values = new_attribute_values
        self.attributes[attribute_name] = attribute_values
    
    def add_attributes_dict(self, attributes_dict):
        for attribute_name, attribute_values in attributes_dict.items():
            self.add_attribute(attribute_name, attribute_values)
    
    
    def load_attributes(self, attributes_path):
        temp_attributes = graphIO.read_attributes(attributes_path)
        for attribute_name, attribute_values in temp_attributes.items():
            if len(attribute_values) != self.size()*2:
                raise ValueError(
                    "The number of attribute values must be equal to the number of regions.")
            self.add_attribute(attribute_name, attribute_values)
    
    def save_attributes(self, attributes_path):
        graphIO.write_attributes(attributes_path, self.attributes)

    def save_significance_matrix(self, p_values, significance_matrix_path):
        if len(p_values) != self.size():
            raise ValueError(
                "The number of p-values must be equal to the number of regions.")
        significance_matrix = np.zeros((self.size()*2, self.size()*2), dtype=float)
        for i, region in enumerate(self.region_names):
            significance_matrix[self.lh_regions[region], self.rh_regions[region]] = 1/p_values[i]
            significance_matrix[self.rh_regions[region], self.lh_regions[region]] = 1/p_values[i]
        graphIO.write_adj_matrix(significance_matrix_path, significance_matrix)



    

    


if __name__ == "__main__":
    regions = graphIO.read_mappings_from_json("data/region_name_mapping.json")
    hemispheres = RegionsHandler(regions)
    
