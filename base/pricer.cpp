#include<iostream> 
#include<cmath> 
#include<random> 
#include<chrono> 

double Get_Normalized_Random(){
    static std :: mt19937 generator(1337); 
    static std :: normal_distribution<double> distribution(0.0 , 1.0); 
    return distribution(generator); 
}
int main(){

    /*
    init stock price 
    strike price 
    risk free intrest 
    colatiliry 
    time to maturity 
    no of sims 
    */
    const double S0 = 100; 
    const double K = 105.0; 
    const double r = 0.05; 
    const double sigma = 0.2; 
    const double T = 1.0; 
    const int N = 10e6 ; 
    double total_payoff = 0.0; 

    auto start_timer = std :: chrono:: high_resolution_clock::now(); 

    for(int i = 0 ; i < N; i++){
        double Z = Get_Normalized_Random(); 

        double ST = S0 * (std :: exp ((r - 0.5 * sigma*sigma)*T + sigma * std :: sqrt(T)*Z)); 
        double payoff =std ::  max((double)0.0 , ST - K) ;
        total_payoff += payoff; 
    }
    double mean_payoff = total_payoff / N ; 
    // multiplying so we adjust for intrest rate 
    double options_price = mean_payoff * std :: exp(-r * T); 
    auto end_time = std :: chrono :: high_resolution_clock :: now(); 
    std :: chrono :: duration <double> duration = end_time - start_timer ; 
    std::cout << "--- Baseline Monte Carlo ---" << std::endl;
    std::cout << "Simulations:    " << N << std::endl;
    std::cout << "Option Price:   " << options_price << std::endl;
    std::cout << "Time Elapsed: " << duration.count() << " seconds" << std::endl;

    return 0 ; 
}