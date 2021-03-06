#pragma once
#include <cstddef> // for size_t on GCC

struct Statistics {
	// network
	size_t n_net_send, n_net_recv;
	size_t b_net_send, b_net_recv;
	double t_net_send, t_net_recv;
	double t_data_serial, t_data_deserial; // part of the net_send and net_recv time
	// delta
	size_t n_dlt_send, n_dlt_recv;
	double t_dlt_calc, t_dlt_wait, t_dlt_send;
	// parameter
	size_t n_par_send, n_par_recv;
	double t_par_calc, t_par_wait, t_par_send;
	// calculation
	size_t n_iter, n_point;
	// other
	double t_data_load, t_train_prepare;

	// summary
	double t_smy_work, t_smy_wait;

	// specific
	size_t n_archive;
	double t_archive;

	Statistics();
};
