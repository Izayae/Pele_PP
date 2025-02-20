#include <HeaderReader.H>

// Class inspired by Olivier's tool mandoline

// Constructor
HeaderReader::HeaderReader(std::string header_path, int max_level=100) : limit_level(max_level)
{
    //std::cout << "\nReading " << header_path << "\n";
    // Open the header file
    std::ifstream header_file(header_path);
    std::string line;
    //std::cout << header_path << std::endl;
    if (header_file.is_open())
    {
        // Read headers information in the right order
        header_file >> header_version;
        header_file >> n_field;
        field_names.resize(n_field);
        for (int i_field=0; i_field<n_field; i_field++) {
            header_file >> field_names[i_field];
        }
        header_file >> n_dim;
        header_file >> sim_time;
        header_file >> finest_level;
        
        domain_lo.resize(n_dim);
        domain_hi.resize(n_dim);
        for (int i_dim=0; i_dim<n_dim; i_dim++) {
            header_file >> domain_lo[i_dim];
        }
        for (int i_dim=0; i_dim<n_dim; i_dim++) {
            header_file >> domain_hi[i_dim];
        }
        ref_ratio.resize(finest_level);
        for (int i_lev=0; i_lev<finest_level; i_lev++) {
            header_file >> ref_ratio[i_lev];
        }
        getline(header_file, line);   // new line
        getline(header_file, line);   // skip grid line too hard too read...
        getline(header_file, line);   // skip this line too cause useless here...
        dx.resize(finest_level+1);
        for (int i_lev=0; i_lev<=finest_level; i_lev++) {
            header_file >> dx[i_lev] >> dx[i_lev] >> dx[i_lev];
        }
        // Instead, reconstruct it with dx and domain size
        grid_level.resize(finest_level+1);
        N_coord.resize(finest_level+1);
        for (int i_lev=0; i_lev<=finest_level; i_lev++) {
            grid_coord grid(n_dim);
            N_coord[i_lev].resize(n_dim);
            for (int i_dim=0; i_dim<n_dim; i_dim++) {
                N_coord[i_lev][i_dim] = round((domain_hi[i_dim]-domain_lo[i_dim])/dx[i_lev]);
                grid.lo[i_dim] = 0;
                grid.hi[i_dim] = N_coord[i_lev][i_dim]-1;
                grid.grow[i_dim] = 0;
            }
            grid_level[i_lev] = grid;
        }
        getline(header_file, line);   // new line
        getline(header_file, line);   // skip this ? line
        getline(header_file, line);   // skip this ? line
        
        // start reading each level boxes (to the level specified in max_level)
        int lev;  // dummy for first line reading
        limit_level = std::min(limit_level, finest_level);
        box_list.resize(limit_level+1);
        n_box.resize(limit_level+1);
        
        for (int i_lev=0; i_lev<=limit_level; i_lev++) {
            header_file >> lev >> n_box[i_lev] >> sim_time; // first line
            getline(header_file, line);   // new line...
            getline(header_file, line);   // skip this line cause useless here...
            box_list[i_lev].resize(n_box[i_lev]);
            for (int i_box=0; i_box<n_box[i_lev]; i_box++) {
                box_coord box(n_dim);
                for (int i_dim=0; i_dim<n_dim; i_dim++) {
                    header_file >> box.real_lo[i_dim] >> box.real_hi[i_dim];
                    // Compute the numerical grid box upper and lower boundaries
                    box.num_lo[i_dim] = round(box.real_lo[i_dim]/dx[i_lev]);
                    box.num_hi[i_dim] = round(box.real_hi[i_dim]/dx[i_lev])-1;
                }
                box_list[i_lev][i_box] = box;
            }
            getline(header_file, line);   // new line...
            getline(header_file, line);   // skip last line with Level_?/Cell
        }    
    } else {
        std::cout << "File not correctly opened\n";
    }
    
    header_file.close();
}

void
HeaderReader::PrintHeader() {
    // print header information for validation and to the limit level specified (skipping useless and cumbersomes lines)
    std::cout << header_version << std::endl;
    std::cout << n_field << std::endl;
    for (int i_field=0; i_field<n_field; i_field++) {
        std::cout << field_names[i_field] << " ";
    }
    std::cout << "\n";
    std::cout << n_dim << std::endl;
    std::cout << sim_time << std::endl;
    std::cout << finest_level << std::endl;
    for (int i_dim=0; i_dim<n_dim; i_dim++) {
        std::cout << domain_lo[i_dim] << " ";
    }
    std::cout << "\n";
    for (int i_dim=0; i_dim<n_dim; i_dim++) {
        std::cout << domain_hi[i_dim] << " ";
    }
    std::cout << "\n";
    for (int i_lev=0; i_lev<limit_level; i_lev++) {
        std::cout << ref_ratio[i_lev] << " ";
    }
    std::cout << "\n";
    for (int i_lev=0; i_lev<=limit_level; i_lev++) {
    
        std::cout << "(";
        for (int i_dim=0; i_dim<n_dim; i_dim++) {
            std::cout << "(";
            std::cout << grid_level[i_lev].lo[i_dim] << ",";
            std::cout << ") ";
        }
        for (int i_dim=0; i_dim<n_dim; i_dim++) {
            std::cout << "(";
            std::cout << grid_level[i_lev].hi[i_dim] << ",";
            std::cout << ") ";
        }
        for (int i_dim=0; i_dim<n_dim; i_dim++) {
            std::cout << "(";
            std::cout << grid_level[i_lev].grow[i_dim] << ",";
            std::cout << ")";
        }
        std::cout << ") ";
    }
    std::cout << "\n";
    for (int i_lev=0; i_lev<=limit_level; i_lev++) {
        std::cout << "dx at level " << i_lev << ": " << dx[i_lev] << "\n";
    }
    for (int i_lev=0; i_lev<=limit_level; i_lev++) {
        std::cout << "Box in level " << i_lev << ": " << n_box[i_lev] << "\n";
    }
}