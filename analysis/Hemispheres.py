import graphIO


class Hemispheres:
    def __init__(self, region_mappings_path, source='dict'):
        if source == 'dict':
            self.region_mappings = graphIO.read_mappings_from_json(
                region_mappings_path)
        elif source == 'node':
            self.region_mappings = graphIO.read_mappings_from_node_file(
                region_mappings_path)
        else:
            raise ValueError(
                "The 'from' parameter must be either 'dict' or 'node'.")
        self.region_mappings = {key - 1: value for key,
                                value in self.region_mappings.items()}
        self.reverse_region_mappings = {
            value: key for key, value in self.region_mappings.items()}
        self.lh_regions = {}
        self.rh_regions = {}
        for key, value in self.region_mappings.items():
            if 'lh' in value:
                self.lh_regions[value.replace('_lh_', '')] = key
            else:
                self.rh_regions[value.replace('_rh_', '')] = key
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

    def size(self):
        return len(self.lh_regions)

    def lh_indices(self):
        return list(self.lh_regions.values())

    def rh_indices(self):
        return list(self.rh_regions.values())

    def region_names(self):
        return list(self.lh_regions.keys())

    def region_index(self, region_name):
        assert list(self.lh_regions.keys()).index(
            region_name) == list(self.rh_regions.keys()).index(region_name)
        return list(self.lh_regions.keys()).index(region_name)

    def lh_region_index(self, region_name):
        if '_lh_' in region_name:
            region_name = region_name.replace('_lh_', '')
        if '_rh_' in region_name:
            region_name = region_name.replace('_rh_', '')
        return self.lh_regions[region_name]

    def rh_region_index(self, region_name):
        if '_lh_' in region_name:
            region_name = region_name.replace('_lh_', '')
        if '_rh_' in region_name:
            region_name = region_name.replace('_rh_', '')
        return self.rh_regions[region_name]


if __name__ == "__main__":
    hemispheres = Hemispheres("data/region_name_mapping.json")
    print(hemispheres.size())
    print(hemispheres.region_names())
    print(list(hemispheres.rh_regions.keys())[-6:])
    print(list(hemispheres.lh_regions.keys())[-6:])
    print(hemispheres.lh_region_index("subamy"))
    print(hemispheres.rh_region_index("subamy"))
    print(hemispheres.region_index("subamy"))
