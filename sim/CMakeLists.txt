add_executable(ps_sim ps_sim.c sim_world.c scene.c nn_ref.c)
target_link_libraries(ps_sim pyrosight_core)
