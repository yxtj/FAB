#include "Statistics.h"

Statistics::Statistics()
	:n_net_send(0), n_net_recv(0),
	b_net_send(0), b_net_recv(0),
	t_net_send(0.0), t_net_recv(0.0),
	t_data_serial(0.0), t_data_deserial(0.0),
	n_dlt_send(0), n_dlt_recv(0),
	t_dlt_calc(0.0), t_dlt_wait(0.0),
	n_par_send(0), n_par_recv(0),
	t_par_calc(0), t_par_wait(0.0),
	n_iter(0), n_point(0),
	t_data_load(0.0), t_train_prepare(0.0),
	t_smy_work(0.0), t_smy_wait(0.0),
	n_archive(0), t_archive(0.0)
{
}
