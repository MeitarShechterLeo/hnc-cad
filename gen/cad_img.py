import argparse
from OCC.Core.Graphic3d import *
from OCC.Display.OCCViewer import Viewer3d
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_WHITE
from OCC.Core.V3d import V3d_DirectionalLight
from OCC.Core.gp import gp_Dir
from glob import glob
import pathlib
from tqdm import tqdm

import subprocess
import os
from contextlib import contextmanager
from OCC.Display.SimpleGui import init_display

viewer = None


@contextmanager
def use_xvfb():
    """
    Context manager to start and stop a virtual X server (Xvfb) for headless operation.
    """
    xvfb_process = None
    if not is_display_active(":1"):
        xvfb_process = subprocess.Popen("Xvfb :1 -screen 0 1024x768x16", shell=True)
    os.environ["DISPLAY"] = ":1"
    try:
        yield
    finally:
        if xvfb_process is not None:
            xvfb_process.terminate()

def is_display_active(display):
    """
    Check if a X server display is active.

    Args:
        display (str): The display identifier.

    Returns:
        bool: True if the display is active, False otherwise.
    """
    # Set the DISPLAY environment variable to the specified display
    os.environ['DISPLAY'] = display

    try:
        # Run xdpyinfo and ignore its output
        subprocess.run(['xdpyinfo'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        # xdpyinfo failed, meaning the display is not active or accessible
        return False

def initialize_display():
    """
    Initialize a display with specific settings.

    Returns:
        Display: The initialized display.
    """
    display, _, _, _ = init_display(backend_str="tk", size=(1024, 768), display_triedron=False, background_gradient_color1=[206, 215, 222], background_gradient_color2=[128,128,128])
    return display


def render(shape, filename, width=1024, height=768, face_color_rgb=(0.2, 0.2, 0.2), edge_color_rgb=(0, 0, 0), show_face_boundary=True):
    # # global viewer
    # # if not is_display_active(":1"):
    # #     subprocess.Popen("Xvfb :1 -screen 0 1024x768x16", shell=True)
    # os.environ["DISPLAY"] = ":1"
    # # if viewer is None:
    # viewer = initialize_display()

    # # display.DisplayShape(shape)
    # # display.FitAll()
    # # display.View.Dump(str(out_dir / name))
    # # display.EraseAll()

    # print('initizalized display')


    viewer = Viewer3d()
    viewer.Create(phong_shading=True, create_default_lights=True)
    viewer.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    viewer.SetModeShaded()
    viewer.hide_triedron()
    viewer.EnableAntiAliasing()
    dir_light = V3d_DirectionalLight(gp_Dir(0, 0.5, -1), Quantity_Color(Quantity_NOC_WHITE))
    dir_light.SetEnabled(True)
    dir_light.SetIntensity(500.0)
    viewer.Viewer.AddLight(dir_light)
    viewer.Viewer.SetLightOn()

    viewer.default_drawer.EnableDrawHiddenLine()
    viewer.default_drawer.SetFaceBoundaryDraw(show_face_boundary)
    ais_context = viewer.GetContext()
    dc = ais_context.DeviationCoefficient()
    da = ais_context.DeviationAngle()
    factor = 10
    ais_context.SetDeviationCoefficient(dc / factor)
    ais_context.SetDeviationAngle(da / factor)
    topexp = TopologyExplorer(shape)
    for face in topexp.faces():
        if face is not None:
            viewer.DisplayShape(face, color=Quantity_Color(*face_color_rgb, Quantity_TOC_RGB))
    for edge in topexp.edges():
        if edge is not None:
            viewer.DisplayShape(edge, color=Quantity_Color(*edge_color_rgb, Quantity_TOC_RGB))
    viewer.FitAll()
    viewer.SetSize(width, height)
    viewer.View.Dump(str(filename))

import pdb
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Input folder of STP/STEP files")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder of PNG files")
    p.add_argument("--width", type=int, default=1024, help="Width of image")
    p.add_argument("--height", type=int, default=768, help="Height of image")

    args = p.parse_args()
    
    files = []
    cad_folders = sorted(glob(args.input_dir+'/*/'))
    for folder in cad_folders:
        input_path = pathlib.Path(folder)
        files += list(input_path.glob("*.step"))
    
    output_path = pathlib.Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # with use_xvfb():
    for fn in tqdm(files):
        shape = read_step_file(str(fn))
        render(shape, output_path.joinpath(fn.stem + ".png"), args.width, args.height)


if __name__ == "__main__":
    main()
