// g++ -O3 -std=c++17 -g pricer_threaded.cpp -o pricer_threaded
#include<iostream> 
#include<vector> 
#include<cmath>
#include<chrono> 
#include<thread> 
#include<random> 


void run_simulation_thread(
    int N_Thread, 
    int thread_id , 
    double S0 , 
    double K , 
    double r , 
    double sigma , 
    double T , 
    double &output_thread 
){
    std :: mt19937 generator(std :: random_device{}() + thread_id ); 
    std :: normal_distribution <double> distribution_normal(0.0 , 1.0) ;
    double total_thread_output = 0.0 ; 

    int i = N_Thread ; 
    double Z ; 
    double ST; 
    double payoff; 

    while(i > 0){
        i--;   
        Z = distribution_normal(generator); 
        ST = S0 * std :: exp((r  - 0.5 * sigma * sigma)*T + sigma * Z * std :: sqrt(T));
        total_thread_output += std :: max(0.0 , ST - K) ;
    }
    output_thread = total_thread_output ; 
}
int main(){

    const double S0 = 100.0; 
    const double K = 105.0 ; 
    const double r = 0.05 ; 
    const double sigma = 0.2 ;
    const double T = 1.0 ;
    const int N = 10e6 ;


    // threading from here onwards 
    const int no_of_Avaliable_threads = std :: thread :: hardware_concurrency(); 
    const int num_pre_threads = N / std :: thread ::hardware_concurrency(); 
    std :: vector<std :: thread >threads(no_of_Avaliable_threads); 
    std :: vector<double >partial_outputs (no_of_Avaliable_threads); 

    // out 
    std::cout << "--- Multi-Threaded Monte Carlo ---" << std::endl;
    std::cout << "Total Simulations: " << N << std::endl;
    std::cout << "Using " << no_of_Avaliable_threads << " threads" << std::endl;
    std::cout << "Sims per thread:   "<<  num_pre_threads << std::endl;


    auto start = std :: chrono :: high_resolution_clock :: now() ;

    for (int i = 0; i < no_of_Avaliable_threads; i++)
    {
        threads[i] = std :: thread(run_simulation_thread , num_pre_threads , i , S0 , K , r , sigma , T , std :: ref(partial_outputs[i]) ); 
    }

    for(std :: thread &it : threads ){
        it.join() ;
    }

    // all threads are run now we combine result for monte carlo simulations 

    double total_payoff = 0.0 ; 
    for(double & it : partial_outputs){
        total_payoff += it ;
    }
    double mean_payoff = total_payoff / (num_pre_threads * no_of_Avaliable_threads) ;
    double option_price = mean_payoff * std :: exp(-r*T);

    auto end = std :: chrono :: high_resolution_clock :: now();
    std :: chrono :: duration<double> duration = end - start ; 


    // fonal output for simulation 
    std::cout << "Option Price:      " << option_price << std::endl;
    std::cout << "Time Elapsed:    " << duration.count() << " seconds" << std::endl;

    return 0 ; 
}
