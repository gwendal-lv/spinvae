---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: splash
classes: wide

---

<link rel="stylesheet" href="assets/css/styles.css">

IEEE ICASSP 2023 submission (under review) supplemental material

[View source code on Github](https://github.com/gwendal-lv/spinvae)

---

For the best audio listening experience, please use Chrome (preferred) or Safari.

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

[^1]: TODO ref1
[^2]: TODO ref2
[^3]: TODO ref3
