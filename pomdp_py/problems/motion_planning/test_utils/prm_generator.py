"""Given an environment, generate N connected PRMs and output to a log folder."""

def main(env_name="maze_3d"):

    if env_name.lower() == "maze_3d":
        pyb_env_gui = Maze3D(init_pos + (0, 0, 0), True, False)
        pyb_env = Maze3D(init_pos + (0, 0, 0), False, False)