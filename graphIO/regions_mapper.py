class RegionsMapper:
    def __init__(self, file_path):
        self.regions = self.load_regions(file_path)
    
    def load_regions(self, file_path):
        with open(file_path, 'r') as file:
            regions = [line.strip() for line in file]
        return regions
    
    def regions_count(self):
        return len(self.regions)
    
    def get_region_by_index(self, index):
        return self.regions[index]
    
    def get_region_index(self, region):
        return self.regions.index(region)
    
    def get_region_by_name(self, name):
        for region in self.regions:
            if name.lower() in region.lower():
                return region
        return None
    
    def get_regions_by_hemisphere(self, hemisphere):
        filtered_regions = []
        for region in self.regions:
            if hemisphere == region[-1]:
                filtered_regions.append(region)
        return filtered_regions
    
    def get_all_regions(self):
        return self.regions
    
    def get_region_by_name_and_hemisphere(self, name, hemisphere):
        filtered_regions = []
        for region in self.regions:
            if name.lower() in region.lower() and hemisphere.lower() in region.lower():
                filtered_regions.append(region)
        return filtered_regions
    


if __name__ == "__main__":
    # Code to be executed when the script is run directly
    regions_mapper = RegionsMapper('C:\\Users\\mosta\\OneDrive - UNCG\\Academics\\CSC 699 - Thesis\\data\\regions\\AAL116.txt')
    print(regions_mapper.regions_count())
    pass