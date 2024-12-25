import click
import dnnlib
import legacy
from utils.Calculators2 import StyleGAN2_FLOPCal



@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
def calculate_flops(
    ctx: click.Context,
    network_pkl: str
):
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to('cpu')
    state_dict = G.state_dict()
    flops = StyleGAN2_FLOPCal(state_dict)
    print('FLOPs: ', flops, ' | ', flops / 1e9 , 'B')
    
if __name__ == "__main__":
    calculate_flops()