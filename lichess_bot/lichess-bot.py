"""Starting point for lichess-bot."""
from lib.lichess_bot import start_program
#import jax


if __name__ == "__main__":
    #jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    #jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    #jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    #jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    start_program()
