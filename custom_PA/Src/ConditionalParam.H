#ifndef CONDITIONALPARAM_H
#define CONDITIONALPARAM_H

#include <AMReX_Vector.H>
#include <AMReX_Array4.H>

struct Condition{
    // Structure for each individual condition
    std::string condition_name;
    std::string type;
    std::string param = "None";
    std::string var;
    // Boundaries of the condition (polynomial form)
    // width is for band type conditions
    amrex::Real width;
    amrex::Vector<amrex::Real> lo;
    amrex::Vector<amrex::Real> hi;
    
    // Constructor for conditions
    Condition(std::string condition_name)
    : condition_name(condition_name)
    {
        // Initialize parser with the name of the condition
        amrex::ParmParse ppc("cond." + condition_name);
        
        // Read condition parameters
        ppc.query("type", type);
        if (type == "constant") {
            ppc.query("var", var);
            // lower boundary
            int n_lo;
            lo.resize(1);
            ppc.getarr("lo", lo);
            
            // higher boundary
            int n_hi;
            lo.resize(1);
            ppc.getarr("hi", hi);
            
        } else if (type == "function") {
            ppc.query("var", var);
            ppc.query("param", param);
            
            // lower boundary
            int n_lo = ppc.countval("lo");
            lo.resize(n_lo);
            ppc.getarr("lo", lo);
            amrex::Print() << "\n";
            // higher boundary
            int n_hi = ppc.countval("hi");
            hi.resize(n_hi);
            ppc.getarr("hi", hi);
        } else if (type == "band") {
            ppc.query("var", var);
            ppc.query("param", param);
            ppc.query("width", width);
            
            // lower and higher boundaries from band centerline
            // half the width is simply added to the constant coefficient of the polynomial
            int n_band = ppc.countval("band");
            lo.resize(n_band);
            hi.resize(n_band);
            ppc.getarr("band", lo);
            ppc.getarr("band", hi);
            lo[0] = lo[0]-width/2;
            hi[0] = hi[0]+width/2;
        } else {
            amrex::Abort("type " + type + "is not accepted for conditioning\n");
        }
    }
    
    // Functions for conditions //
    int
    getIdVar(const std::string name_var, const amrex::Vector<std::string> list_name){
        // get index of the position of a string in a given list of string
        // output error if it's not loaded
        int id_var;
        auto itr = std::find(list_name.begin(), list_name.end(), name_var);
        if (itr != list_name.cend()){
            id_var = std::distance(list_name.begin(), itr);
        } else {
            std::cout << "Condition variable " << name_var << " not loaded" << "\n";
            amrex::Error();
        }
        return id_var;
    }
    
    std::string
    getConditionStr(){
        std::ostringstream oss;
        oss << condition_name << " of type " << type << "\n";
        oss << "Y=" << var << ", X=" << param << "\n";
        oss << "lower bound: Y(X) = " << lo[0];
        for (int ic=1; ic<lo.size(); ic++) {
            oss << " + " << lo[ic] << ".X^" << ic;
        }
        oss << "\n";
        oss << "higher bound: Y(X) = " << hi[0];
        for (int ic=1; ic<hi.size(); ic++) {
            oss << " + " << hi[ic] << ".X^" << ic;
        }
        oss << "\n";
        std::string condition_str = oss.str();
        return condition_str;
    }
    
    void
    printCondition(){
        std::string condition_str = getConditionStr();
        amrex::Print() << condition_str;
    }
    
    int
    checkCondition(const int i, const int j, const int k,
                   const amrex::Array4<const amrex::Real> fab, const amrex::Vector<std::string> list_name) {
        // Check if this condition is satisfied
        int contribute = 1;
        int id_var = getIdVar(var, list_name);
        amrex::Real c_val = fab(i, j, k, id_var);
        int id_param;
        amrex::Real lo_val = lo[0];
        amrex::Real hi_val = hi[0];
        if (param != "None") {
            int id_param = getIdVar(param, list_name);
            amrex::Real p_val = fab(i, j, k, id_param);
            for (int ic=1; ic<lo.size(); ic++) {
                lo_val += lo[ic]*std::pow(p_val, ic);
            }
            for (int ic=1; ic<hi.size(); ic++) {
                hi_val += hi[ic]*std::pow(p_val, ic);
            }
        }
        if (c_val<lo_val || c_val>hi_val) contribute=0;
        
        return contribute;
    }
};

typedef struct ConditionalParam{
    int do_conditioning = 0;
    int n_cond = 0;
    amrex::Vector<Condition> condition_container;
    
    // Constructor for conditions
    ConditionalParam() {
        // Parameter parser
        amrex::ParmParse pp("cond");
        pp.query("do_conditioning", do_conditioning);
        
        // Filling structure
        if (do_conditioning){
            // Number of conditions and resize the condition container
            n_cond = pp.countval("condition_names");
            
            // Read the conditions one by one
            // And create the Condition object in the container
            std::string condition_name;
            for (int i_cond=0; i_cond<n_cond; i_cond++) { 
                pp.get("condition_names", condition_name, i_cond);
                Condition single_condition(condition_name);
                condition_container.push_back(single_condition);
            }
        }
    }
    
    // Utility function
    void
    printConditions(){
        if (do_conditioning){
            for (int i_cond=0; i_cond<n_cond; i_cond++) {
                amrex::Print() << "Conditioning n°" << i_cond << ":\n";
                condition_container[i_cond].printCondition();
            }
        } else {
            amrex::Print() << "Conditioning is not activated\n";
        }
    }
    
    int
    checkConditions(const int i, const int j, const int k,
                    const amrex::Array4<const amrex::Real> fab, const amrex::Vector<std::string> list_name) {
        // Check if the user-specified conditions are satisfied
        int contribute = 1;
        for (int i_cond=0; i_cond<n_cond; i_cond++) {
            // If one of the condition is not verified contribution of this datapoint is null
            if (!condition_container[i_cond].checkCondition(i, j, k, fab, list_name)) {
                contribute=0;
            }
        }
        
        return contribute;
    }
    
    // Write conditions for this run in a metadata txt file 
    void
    writeConditions(std::string path_final) {
        if (amrex::ParallelDescriptor::IOProcessor()){
            // Create the directory tree for the extracted data if it doesn't exist
            amrex::Print() << "Writing conditions in " << path_final << std::endl;
            const int dir_err = std::system(("mkdir -p " + path_final).c_str());
            if (-1 == dir_err)
            {
                printf("Error creating directory!n");
                exit(1);
            }

            // Create a separate header file for metadata
            //Open and write the file
            std::string header_name = path_final + "/conditions.txt";
            FILE *file;
            file = fopen((header_name).c_str(),"w");
            // Check that the file is correctly opened
            if (file != NULL)
            {
                //std::cout << "Header file successfully open\n";
            } else {
                std::cout << "Condition file not opened, crash incoming\n";
                std::cout << (header_name).c_str() << "\n";
                perror("fopen");
            }

            // Write in this file the conditions
            for (int i_cond=0; i_cond<n_cond; i_cond++) {
                fprintf(file, condition_container[i_cond].getConditionStr().c_str());
                condition_container[i_cond].printCondition();
            }

            fclose(file);
        } // end IOProcess
    }
} ConditionalParam;

#endif