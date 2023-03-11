---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: splash
classes: wide

---

<link rel="stylesheet" href="assets/css/styles.css">

*Authors: Gwendal Le Vaillant and Thierry Dutoit ([ISIA Lab](https://web.umons.ac.be/isia/), University of Mons)*

This page contains supplemental material for the paper "Synthesizer Preset Interpolation using Transformer Auto-Encoders" accepted to IEEE ICASSP 2023 (preprint: [https://arxiv.org/abs/2210.16984](https://arxiv.org/abs/2210.16984)).

For the best audio listening experience, please use Chrome (preferred) or Safari.

Contents:

- [Interpolation between presets](#interpolation-between-presets)
- [Extrapolation](#extrapolation)
- [Audio features and interpolation performance results](#audio-features-and-interpolation-performance-results)

[Source code is available on Github](https://github.com/gwendal-lv/spinvae)

---

# Interpolation between presets


<!--
TODO describe methods

TODO describe [Dexed](https://asb2m10.github.io/dexed/)

- The "latent space" interpolation consists in encoding presets into the latent space using TODO describe
Then, a linear interpolation is performed on latent vectors TODO describe transformer decode

- The "naive" interpolation consists in a linear interpolation between VST parameters TODO improve description
-->

### Interpolation example 1

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"WindEns2Ed"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"HARD ROADS"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/9</td>
            <td>Step 2/9</td>
            <td>Step 3/9</td>
            <td>Step 4/9</td>
            <td>Step 5/9</td>
            <td>Step 6/9</td>
            <td>Step 7/9</td>
            <td>Step 8/9</td>
            <td>Step 9/9</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">SPINVAE</th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00135spinvae_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135spinvae_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135spinvae_spectrogram_step08.png"/>
            </td>
        </tr>
        <tr> <!-- naive interp -->
            <th scope="row">
                Reference <br  /> (naive linear)
            </th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00135reflinear_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00135reflinear_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00135reflinear_spectrogram_step08.png"/>
            </td>
        </tr>
    </table>
</div>

### Interpolation example 2

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"BRASS 15"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"Kharma'HM"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/9</td>
            <td>Step 2/9</td>
            <td>Step 3/9</td>
            <td>Step 4/9</td>
            <td>Step 5/9</td>
            <td>Step 6/9</td>
            <td>Step 7/9</td>
            <td>Step 8/9</td>
            <td>Step 9/9</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">SPINVAE</th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00034spinvae_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034spinvae_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034spinvae_spectrogram_step08.png"/>
            </td>
        </tr>
        <tr> <!-- naive interp -->
            <th scope="row">
                Reference <br  /> (naive linear)
            </th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00034reflinear_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00034reflinear_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00034reflinear_spectrogram_step08.png"/>
            </td>
        </tr>
    </table>
</div>

### Interpolation example 3

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"VIBESYN 2"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"SYNTHEKLA4"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/9</td>
            <td>Step 2/9</td>
            <td>Step 3/9</td>
            <td>Step 4/9</td>
            <td>Step 5/9</td>
            <td>Step 6/9</td>
            <td>Step 7/9</td>
            <td>Step 8/9</td>
            <td>Step 9/9</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">SPINVAE</th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00010spinvae_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010spinvae_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010spinvae_spectrogram_step08.png"/>
            </td>
        </tr>
        <tr> <!-- naive interp -->
            <th scope="row">
                Reference <br  /> (naive linear)
            </th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00010reflinear_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00010reflinear_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00010reflinear_spectrogram_step08.png"/>
            </td>
        </tr>
    </table>
</div>


### Interpolation example 4

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"Synbass 4"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"K.CLAV. 3"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/9</td>
            <td>Step 2/9</td>
            <td>Step 3/9</td>
            <td>Step 4/9</td>
            <td>Step 5/9</td>
            <td>Step 6/9</td>
            <td>Step 7/9</td>
            <td>Step 8/9</td>
            <td>Step 9/9</td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <th scope="row">SPINVAE</th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00018spinvae_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018spinvae_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018spinvae_spectrogram_step08.png"/>
            </td>
        </tr>
        <tr> <!-- naive interp -->
            <th scope="row">
                Reference <br  /> (naive linear)
            </th>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step00.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/interpolation/00018reflinear_spectrogram_step00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step06.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step06.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step07.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step07.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/interpolation/00018reflinear_audio_step08.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/interpolation/00018reflinear_spectrogram_step08.png"/>
            </td>
        </tr>
    </table>
</div>

---

# Extrapolation

### SPINVAE extrapolation example 1

<div class="figure">
    <table>
        <tr>
            <th colspan="2" class=centered_th><--- Extrapolation</th>
            <th>Preset</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>Preset</th>
            <th colspan="2" class=centered_th>Extrapolation ---></th>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td>"PnoCk Ep9"</td>
            <td colspan="4" class=centered_th><---------- Interpolation ----------></td>
            <td>"SYNTH 7"</td>
            <td></td>
            <td></td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_extrap_m2.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/extrapolation/080100to074548_spectrogram_extrap_m2.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_extrap_m1.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_extrap_m1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_interp_00.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_interp_00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_interp_01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_interp_01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_interp_02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_interp_02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_interp_03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_interp_03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_interp_04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_interp_04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_interp_05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_interp_05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_extrap_p1.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_extrap_p1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/080100to074548_audio_extrap_p2.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/080100to074548_spectrogram_extrap_p2.png"/>
            </td>
        </tr>
    </table>
</div>


### SPINVAE extrapolation example 2

<div class="figure">
    <table>
        <tr>
            <th colspan="2" class=centered_th><--- Extrapolation</th>
            <th>Preset</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>Preset</th>
            <th colspan="2" class=centered_th>Extrapolation ---></th>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td>"BOUM"</td>
            <td colspan="4" class=centered_th><---------- Interpolation ----------></td>
            <td>"fuzzerro"</td>
            <td></td>
            <td></td>
        </tr>
        <tr> <!-- SPINVAE interp -->
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_extrap_m2.mp3" type="audio/mp3" /></audio><br />
                <img src="assets/extrapolation/199874to016527_spectrogram_extrap_m2.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_extrap_m1.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_extrap_m1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_interp_00.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_interp_00.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_interp_01.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_interp_01.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_interp_02.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_interp_02.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_interp_03.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_interp_03.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_interp_04.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_interp_04.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_interp_05.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_interp_05.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_extrap_p1.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_extrap_p1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control><source src="assets/extrapolation/199874to016527_audio_extrap_p2.mp3" type="audio/mp3" /></audio><br/>
                <img src="assets/extrapolation/199874to016527_spectrogram_extrap_p2.png"/>
            </td>
        </tr>
    </table>
</div>

---

# Audio features and interpolation performance results

### Timbre audio features

As indicated in the paper, the Timbre Toolbox [^1] is used to compute audio features for each rendered audio file. 
These features have been engineered by experts and span a wide range of timbre characteristics. They can be categorized into three main groups as shown in the table below.

| | Temporal | | Spectral | | Harmonic |
|--:|---|--:|---|--:|---|
| *Att<br>Dec<br>Rel<br>LAT<br>AttSlope<br>DecSlope<br>TempCent<br>EffDur<br>FreqMod<br>AmpMod<br>RMSEnv* | Attack<br>Decay<br>Release<br>Log-Attack Time<br>Attack Slope<br>Decrease Slope<br>Temporal Centroid<br>Effective Duration<br>Frequency of Energy Mod.<br>Amplitude of Energy Mod.<br>RMS of Energy Envelope |  *SpecCent<br>SpecSpread<br>SpecSkew<br>SpecKurt<br>SpecSlope<br>SpecDecr<br>SpecRollOff<br>SpecVar<br>FrameErg<br>SpecFlat<br>SpecCrest* |  Spectral Centroid<br>Spectral Spread<br>Spectral Skewness<br>Spectral Kurtosis<br>Spectral Slope<br>Spectral Decrease<br>Spectral Rolloff<br>Spectro-temporal variation<br>Frame Energy<br>Spectral Flatness<br>Spectral Crest | *HarmErg<br>NoiseErg<br>F0<br>InHarm<br>HarmDev<br>OddEvenRatio*  | Harmonic Energy<br>Noise Energy<br>Fundamental Frequency<br>Inharmonicity<br>Harmonic Spectral Deviation<br>Odd to even harmonic ratio | 

Some features are computed for each time frame, and their median value and Inter-Quartile Range (IQR) only are used[^1]. This results in 46 features available to evaluate the interpolation.

Figure A below shows the correlation between all timbre features, and we observe that most of them are not or weakly correlated. 
Nonetheless, some pairs of features tend to show similar variations (e.g. *RMSEnv_med* and *EffDur*, *SpecCent_med* and *SpecSpread_med*) and present moderate to high levels of correlation.
However, such pairs of features often describe timbre characteristics that are easily distinguishable to the human ear.
E.g., *SpecCent_med* and *SpecSpread_med* are highly correlated (0.72) but are discernibly distinct from each other: a high-pitched sound would have a low Spectral Spread but a high Spectral Centroid. Thus, all features which present a moderate correlation should be used to analyze results.

| Figure A. Absolute value of the correlation between timbre features. | Figure B. Features with a very high (> 0.9) correlation. | 
|---|---|
| <img src="assets/figures/timbre_features_corr.svg"> | <img src="assets/figures/timbre_features_high_corr.svg">  |

However, we observe from Figure B that a few pairs of features present a very high correlation and might be redundant. 
In order to keep only non-highly-correlated features, eight features could be removed from the analysis: *SpecRollOff_med, SpecKurt_med, SpecKurt_IQR, HarmErg_IQR, NoiseErg_med, NoiseErg_IQR, OddEvenRatio_IQR* and *Rel*. This is discussed in the next sub-section.


### Detailed interpolation results

Interpolation quality is evaluated using the smoothness and non-linearity of audio features along an interpolation sequence. More details about these interpolation metrics are available in the paper.

Results from Table 1 from the paper are reproduced on the left-hand side of the table below.
They use the full set of 46 audio features, and include the number of improved features and average performance variation.

Additional results are presented on the right-hand side of the table below. They show the median performance variation for 46 features (not presented in the paper due to space constraints).
They also include complete results (number of improved features, average and median performance variaton) obtained using the reduced set of 46-8 = 38 features.

SPINVAE remains the best overall model, with no significant difference in results between using the full or reduced set of audio features.

<div class="figure">
    <table class=centered_th>
        <tr>
            <th></th>
            <th colspan="4" class=centered_th style="background-color:lightgoldenrodyellow;">Results included in the paper</th>
            <th colspan="9" class=centered_th style="background-color:lightgreen;">Additional results</th>
        </tr>
        <tr>
            <th rowspan="3">Model</th>
            <th colspan="6" class=centered_th>46 features</th>
            <th>|</th>
            <th colspan="6" class=centered_th>38 features (highly correlated excluded)</th>
        </tr>
        <tr>
            <th colspan="2" class=centered_th>Num. improved<br>features (out of 46)</th>
            <th colspan="2" class=centered_th>Average<br>variation (%)</th>
            <th colspan="2" class=centered_th>Median<br>variation (%)</th>
            <th>|</th>
            <th colspan="2" class=centered_th>Num. improved<br>features (out of 38)</th>
            <th colspan="2" class=centered_th>Average<br>variation (%)</th>
            <th colspan="2" class=centered_th>Median<br>variation (%)</th>
        </tr>
        <tr>
            <th>Smooth.</th>
            <th>Nonlin.</th>
            <th>Smooth.</th>
            <th>Nonlin.</th>
            <th>Smooth.</th>
            <th>Nonlin.</th>
            <th>|</th>
            <th>Smooth.</th>
            <th>Nonlin.</th>
            <th>Smooth.</th>
            <th>Nonlin.</th>
            <th>Smooth.</th>
            <th>Nonlin.</th>
        </tr>
        <tr>
            <td class="bestresult">SPINVAE (DLM 3)</td>
            <td class="bestresult"><span>35</span></td>
            <td>38</td>
            <td class="bestresult">-12.6</td>
            <td>-12.3</td>
            <td class="bestresult">-13.1</td>
            <td>-15.5</td>
            <td>|</td>
            <td class="bestresult">29</td>
            <td>30</td>
            <td class="bestresult">-12.1</td>
            <td>-11.3</td>
            <td class="bestresult">-12.0</td>
            <td>-14.2</td>
        </tr>
        <tr>
            <td>Preset-only</td>
            <td>25</td>
            <td>30</td>
            <td>-4.6</td>
            <td>-6.4</td>
            <td>-4.6</td>
            <td>-9.0</td>
            <td>|</td>
            <td>20</td>
            <td>22</td>
            <td>-3.9</td>
            <td>-5.8</td>
            <td>-4.1</td>
            <td>-7.5</td>
        </tr>
        <tr>
            <td>Sound matching</td>
            <td>8</td>
            <td>7</td>
            <td>+66.8</td>
            <td>+29.7</td>
            <td>+156</td>
            <td>+109</td>
            <td>|</td>
            <td>6</td>
            <td>5</td>
            <td>+62.4</td>
            <td>+28.8</td>
            <td>+172</td>
            <td>+126</td>
        </tr>
        <tr>
            <td>DLM 2</td>
            <td>31</td>
            <td>37</td>
            <td>-8.2</td>
            <td>-10.4</td>
            <td>-6.7</td>
            <td>-11.0</td>
            <td>|</td>
            <td>26</td>
            <td>29</td>
            <td>-7.4</td>
            <td>-9.3</td>
            <td>-5.6</td>
            <td>-9.5</td>
        </tr>
        <tr>
            <td>DLM 4</td>
            <td>30</td>
            <td class="bestresult">40</td>
            <td>-9.2</td>
            <td>-14.5</td>
            <td>-7.5</td>
            <td class="bestresult">-16.1</td>
            <td>|</td>
            <td>25</td>
            <td class="bestresult">32</td>
            <td>-9.2</td>
            <td>-13.7</td>
            <td>-7.0</td>
            <td class="bestresult">-15.0</td>
        </tr>
        <tr>
            <td>Softmax</td>
            <td>23</td>
            <td class="bestresult">40</td>
            <td>-1.2</td>
            <td class="bestresult">-15.6</td>
            <td>+0.9</td>
            <td>-15.9</td>
            <td>|</td>
            <td>18</td>
            <td class="bestresult">32</td>
            <td>-1.4</td>
            <td class="bestresult">-14.6</td>
            <td>+2.5</td>
            <td>-13.5</td>
        </tr>
        <tr>
            <td>MLP</td>
            <td>18</td>
            <td>27</td>
            <td>+21.0</td>
            <td>-1.8</td>
            <td>+23.9</td>
            <td>+0.3</td>
            <td>|</td>
            <td>15</td>
            <td>22</td>
            <td>+18.2</td>
            <td>-1.5</td>
            <td>+20.7</td>
            <td>+3.4</td>
        </tr>
        <tr>
            <td>LSTM</td>
            <td>15</td>
            <td>1</td>
            <td>+123</td>
            <td>+93.5</td>
            <td>+349</td>
            <td>+394</td>
            <td>|</td>
            <td>12</td>
            <td>1</td>
            <td>+109</td>
            <td>+89.5</td>
            <td>+403</td>
            <td>+462</td>
        </tr>
    </table>
</div>


---

[^1]: Geoffroy Peeters, Bruno Giordano, Patrick Susini, and Nicolas Misdariis, “The timbre toolbox: Extracting audio descriptors from musical signals,” in The Journal of the Acoustical Society of America, 2011, vol. 130.

