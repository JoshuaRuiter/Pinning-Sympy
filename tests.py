from utility_roots import directed_dynkin_graphs, visualize_graph
from root_system import root_system

def main():
    test_dynkin()

def test_dynkin():
    for name in directed_dynkin_graphs:
        print("\nName:",name)

        if 'BC' in name:
            true_type = 'BC'
        else:
            true_type = name[0]
        true_rank = int(name[-1])
        print("\tTrue type:",true_type)
        print("\tTrue rank:",true_rank)
        
        graph = directed_dynkin_graphs[name]
        print("\tGraph visualization:", visualize_graph(graph))
        print("\tGraph as dictionary:",graph)
        calculated_type, calculated_rank = root_system.determine_dynkin_type(graph)
        print("\tDetermined type:",calculated_type)
        print("\tDetermined rank:",calculated_rank)
        
        if type(calculated_type) == str:
            assert(true_type == calculated_type)
        if type(calculated_type) == list:
            assert(true_type in calculated_type)
        
        if type(calculated_rank) == int:
            assert(true_rank == calculated_rank)
        if type(calculated_rank) == list:
            assert(true_rank in calculated_rank)
        
if __name__ == "__main__":
    main()