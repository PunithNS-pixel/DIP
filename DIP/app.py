import io
import numpy as np
import streamlit as st
from PIL import Image
import torch

import final_blind_dip as model


def prepare_uploaded_image(uploaded_file):
    img = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]

    scale = min(1.0, model.PROCESS_LONG_SIDE / max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    if (nh, nw) != (h, w):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def tensor_to_pil(x):
    x = x.detach().cpu().clamp(0, 1)[0]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def run_blind_denoising(noisy_cpu, seed=42, iterations=2000, progress_cb=None):
    model.seed_all(seed)
    noisy = noisy_cpu.to(model.DEVICE)
    target_size = tuple(noisy.shape[-2:])
    h, w = target_size

    train_mask, val_mask = model.create_train_val_masks(seed + 999, h, w)
    median_prior = model.median_baseline(noisy).to(model.DEVICE)

    dip = model.FinalDIP().to(model.DEVICE)
    refiner = model.DetailRefiner().to(model.DEVICE)
    z = torch.rand(
        1, model.DIP_IN_CHANNELS, model.DIP_NOISE_SIZE,
        model.DIP_NOISE_SIZE, device=model.DEVICE
    )

    optimizer = torch.optim.Adam(
        list(dip.parameters()) + list(refiner.parameters()),
        lr=model.LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=model.LEARNING_RATE * 0.10
    )

    best_val = float("inf")
    best_output = None
    no_improvement = 0
    history = []

    for iteration in range(1, iterations + 1):
        dip.train()
        refiner.train()
        optimizer.zero_grad(set_to_none=True)

        base = dip(z, target_size)
        hidden = model.random_refine_mask(
            seed * 100000 + iteration, h, w, train_mask
        )
        masked_noisy = model.blind_input(noisy, hidden, median_prior)
        residual = refiner(masked_noisy, base, hidden)
        output = (base + residual).clamp(0.0, 1.0)

        refine_loss = model.masked_charbonnier(
            output, noisy, hidden.expand_as(noisy)
        )
        base_loss = model.masked_charbonnier(
            base, noisy, train_mask.expand_as(noisy)
        )
        tv = model.total_variation(output)
        loss = 0.35 * base_loss + 0.65 * refine_loss + model.TV_WEIGHT * tv

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dip.parameters()) + list(refiner.parameters()),
            model.GRAD_CLIP,
        )
        optimizer.step()
        scheduler.step()

        dip.eval()
        refiner.eval()
        with torch.no_grad():
            base_eval = dip(z, target_size)
            val_input = model.blind_input(noisy, val_mask, median_prior)
            val_residual = refiner(val_input, base_eval, val_mask)
            eval_out = (base_eval + val_residual).clamp(0.0, 1.0)
            val_loss = model.masked_mse(
                eval_out, noisy, val_mask.expand_as(noisy)
            ).item()

        history.append(val_loss)
        smooth = model.moving_average(history, model.VAL_SMOOTH_WINDOW)

        if smooth < best_val - model.MIN_DELTA:
            best_val = smooth
            best_output = eval_out.detach().cpu().clone()
            no_improvement = 0
        else:
            no_improvement += 1

        if progress_cb and (iteration == 1 or iteration % 10 == 0):
            progress_cb(iteration, iterations, val_loss)

        if iteration >= max(800, model.VAL_SMOOTH_WINDOW) and \
                no_improvement >= model.PATIENCE:
            break

    if best_output is None:
        raise RuntimeError("Blind optimization did not produce a reconstruction.")

    return best_output, iteration


st.set_page_config(
    page_title="Hybrid Blind DIP",
    page_icon="🖼️",
    layout="wide",
)

st.title("Hybrid Blind DIP Image Denoiser")
st.caption(
    "Self-supervised image denoising using a Deep Image Prior + blind detail refiner."
)

with st.sidebar:
    st.header("Settings")
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    iterations = st.slider(
        "Optimization iterations",
        min_value=500,
        max_value=6000,
        value=2000,
        step=100,
    )
    st.info(
        f"Device: {model.DEVICE}\n\n"
        "Production inference uses blind validation only; "
        "no clean reference image is required."
    )

uploaded = st.file_uploader(
    "Upload a noisy image",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded:
    noisy = prepare_uploaded_image(uploaded)

    left, right = st.columns(2)

    with left:
        st.subheader("Input")
        st.image(tensor_to_pil(noisy), use_container_width=True)

    if st.button("Denoise image", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        def update_progress(i, total, val):
            progress.progress(min(i / total, 1.0))
            status.write(
                f"Optimizing: {i}/{total} iterations · "
                f"blind validation loss {val:.6f}"
            )

        with st.spinner("Running blind DIP optimization..."):
            output, completed = run_blind_denoising(
                noisy,
                seed=int(seed),
                iterations=int(iterations),
                progress_cb=update_progress,
            )

        progress.progress(1.0)
        status.success(f"Completed after {completed} iterations.")

        with right:
            st.subheader("Denoised")
            st.image(tensor_to_pil(output), use_container_width=True)

        buf = io.BytesIO()
        tensor_to_pil(output).save(buf, format="PNG")

        st.download_button(
            "Download denoised PNG",
            data=buf.getvalue(),
            file_name="denoised.png",
            mime="image/png",
            use_container_width=True,
        )

        st.caption(
            "Oracle selection is not used in deployment because it requires "
            "the unavailable clean reference."
        )
